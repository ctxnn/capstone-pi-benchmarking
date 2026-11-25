"""I2C access shared by every chip on the cane.

Everything that talks I2C goes through one bus object. The Adafruit
``TCA9548A_Channel`` holds the bus lock for the whole of each transaction
(select the channel, do the transfer, deselect), so the range thread and the
haptic thread can never switch the multiplexer under each other.

The chip drivers in this package (MPU-6050, DRV2605L) are written against the
tiny :class:`RegisterDevice` interface instead of a vendor library, so they can
be unit-tested with an in-memory fake and need no extra dependency.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol


class HardwareUnavailable(RuntimeError):
    """Raised when the Pi hardware libraries or devices cannot be reached."""


class RegisterDevice(Protocol):
    """A chip addressed by 8-bit registers, with its bus locking handled inside."""

    address: int

    def write_reg(self, reg: int, value: int) -> None: ...

    def read_reg(self, reg: int) -> int: ...

    def read_regs(self, reg: int, length: int) -> bytes: ...


class I2CBus(Protocol):
    def channel(self, number: int) -> Any:
        """Return an I2C-like object that selects mux channel ``number`` on use."""

    def register_device(self, address: int, mux_channel: Optional[int]) -> RegisterDevice: ...

    def scan(self) -> Dict[Optional[int], List[int]]:
        """Addresses seen on the main bus (key None) and on each mux channel."""

    def close(self) -> None: ...


class _LockedRegisterDevice:
    """Register access built on ``adafruit_bus_device.I2CDevice``."""

    def __init__(self, i2c_like: Any, address: int) -> None:
        try:
            from adafruit_bus_device.i2c_device import I2CDevice
        except ImportError as exc:  # pragma: no cover - needs the Pi libraries
            raise HardwareUnavailable(
                "adafruit-circuitpython-busdevice is not installed; "
                "run: uv pip install -r requirements-cane-hardware.txt"
            ) from exc
        self.address = address
        self._device = I2CDevice(i2c_like, address, probe=False)

    def write_reg(self, reg: int, value: int) -> None:
        with self._device:
            self._device.write(bytes([reg & 0xFF, value & 0xFF]))

    def read_reg(self, reg: int) -> int:
        return self.read_regs(reg, 1)[0]

    def read_regs(self, reg: int, length: int) -> bytes:
        buffer = bytearray(length)
        with self._device:
            self._device.write_then_readinto(bytes([reg & 0xFF]), buffer)
        return bytes(buffer)


class BlinkaBus:
    """The Pi's I2C bus (GPIO2/GPIO3) with a TCA9548A multiplexer behind it."""

    def __init__(
        self,
        mux_address: int = 0x70,
        *,
        i2c: Any = None,
        mux: Any = None,
    ) -> None:
        self.mux_address = mux_address
        self._i2c = i2c
        self._mux = mux
        self._channels: Dict[int, Any] = {}
        if self._i2c is None:
            self._i2c = self._open_i2c()
        if self._mux is None:
            self._mux = self._open_mux()

    @staticmethod
    def _open_i2c() -> Any:  # pragma: no cover - needs a Pi
        try:
            import board
            import busio
        except ImportError as exc:
            raise HardwareUnavailable(
                "Adafruit Blinka is not installed. Install the cane hardware "
                "requirements and enable I2C (raspi-config -> Interface Options)."
            ) from exc
        except (NotImplementedError, RuntimeError) as exc:
            raise HardwareUnavailable(f"Blinka could not identify this board: {exc}") from exc
        try:
            return busio.I2C(board.SCL, board.SDA)
        except Exception as exc:
            raise HardwareUnavailable(
                f"could not open the I2C bus (is I2C enabled and is the user in the 'i2c' group?): {exc}"
            ) from exc

    def _open_mux(self) -> Any:  # pragma: no cover - needs a Pi
        try:
            import adafruit_tca9548a
        except ImportError as exc:
            raise HardwareUnavailable(
                "adafruit-circuitpython-tca9548a is not installed"
            ) from exc
        try:
            return adafruit_tca9548a.TCA9548A(self._i2c, address=self.mux_address)
        except Exception as exc:
            raise HardwareUnavailable(
                f"no TCA9548A answered at {hex(self.mux_address)}: {exc}"
            ) from exc

    @property
    def main(self) -> Any:
        return self._i2c

    def channel(self, number: int) -> Any:
        if not 0 <= number <= 7:
            raise ValueError("TCA9548A channel must be 0..7")
        if number not in self._channels:
            self._channels[number] = self._mux[number]
        return self._channels[number]

    def register_device(self, address: int, mux_channel: Optional[int]) -> RegisterDevice:
        target = self._i2c if mux_channel is None else self.channel(mux_channel)
        return _LockedRegisterDevice(target, address)

    def scan(self) -> Dict[Optional[int], List[int]]:  # pragma: no cover - needs a Pi
        found: Dict[Optional[int], List[int]] = {}
        while not self._i2c.try_lock():
            pass
        try:
            found[None] = sorted(self._i2c.scan())
            self._i2c.writeto(self.mux_address, b"\x00")
        finally:
            self._i2c.unlock()
        for number in range(8):
            channel = self.channel(number)
            while not channel.try_lock():
                pass
            try:
                found[number] = sorted(
                    address
                    for address in channel.scan()
                    if address != self.mux_address
                )
            finally:
                channel.unlock()
        return found

    def close(self) -> None:
        try:
            while not self._i2c.try_lock():
                pass
            try:
                self._i2c.writeto(self.mux_address, b"\x00")
            finally:
                self._i2c.unlock()
        except Exception:
            pass
        deinit = getattr(self._i2c, "deinit", None)
        if deinit is not None:
            try:
                deinit()
            except Exception:
                pass
