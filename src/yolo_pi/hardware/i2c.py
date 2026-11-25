"""I2C transport, TCA9548A multiplexing, and a register-level test double.

The three VL53L0X sensors and three DRV2605L drivers all answer on the same
addresses (0x29 and 0x5A), so every access has to be serialized through one
TCA9548A channel selection. A single re-entrant lock covers channel selection
plus the transaction itself: without that, one thread could flip the mux
between another thread's select and its register read, and a time-of-flight
reading would silently arrive from the wrong sector.
"""

from __future__ import annotations

import errno
import threading
import time
from typing import Any, ContextManager, Dict, List, Mapping, Optional, Protocol, Tuple

TCA9548A_CONTROL_REGISTER = 0x00
TCA9548A_CHANNEL_COUNT = 8

class SMBusProtocol(Protocol):
    """The subset of ``smbus2.SMBus`` these drivers rely on."""

    def read_byte_data(self, address: int, register: int) -> int:
        ...

    def write_byte_data(self, address: int, register: int, value: int) -> None:
        ...

    def read_i2c_block_data(self, address: int, register: int, length: int) -> List[int]:
        ...

    def write_byte(self, address: int, value: int) -> None:
        ...

    def close(self) -> None:
        ...


class SMBus2Adapter:
    """Thin wrapper around ``smbus2.SMBus`` with a lazy import."""

    def __init__(self, bus_num: int = 1, bus: Any = None) -> None:
        self.bus_num = bus_num
        if bus is None:
            try:
                from smbus2 import SMBus
            except ImportError as exc:  # pragma: no cover - exercised on a Pi
                raise RuntimeError(
                    "smbus2 is required for Raspberry Pi I2C access; "
                    "install it with 'sudo apt install python3-smbus2' or 'pip install smbus2'"
                ) from exc
            bus = SMBus(bus_num)
        self._bus = bus

    def read_byte_data(self, address: int, register: int) -> int:
        return int(self._bus.read_byte_data(address, register))

    def write_byte_data(self, address: int, register: int, value: int) -> None:
        self._bus.write_byte_data(address, register, value & 0xFF)

    def read_i2c_block_data(self, address: int, register: int, length: int) -> List[int]:
        return [int(value) & 0xFF for value in self._bus.read_i2c_block_data(address, register, length)]

    def write_byte(self, address: int, value: int) -> None:
        self._bus.write_byte(address, value & 0xFF)

    def close(self) -> None:
        self._bus.close()


class FakeDevice:
    """One simulated I2C slave with a mutable register map."""

    def __init__(self, address: int, registers: Optional[Mapping[int, int]] = None) -> None:
        self.address = address
        self.registers: Dict[int, int] = {
            int(register) & 0xFF: int(value) & 0xFF for register, value in (registers or {}).items()
        }
        #: Every register written since construction, in order.
        self.write_log: List[Tuple[int, int]] = []
        #: Every register read since construction, in order.
        self.read_log: List[int] = []
        #: Values written with ``write_byte`` (register-less command writes).
        self.command_log: List[int] = []

    def read_register(self, register: int) -> int:
        self.read_log.append(register)
        return self.registers.get(register & 0xFF, 0)

    def write_register(self, register: int, value: int) -> None:
        register &= 0xFF
        value &= 0xFF
        self.write_log.append((register, value))
        self.registers[register] = value

    def read_block(self, register: int, length: int) -> List[int]:
        return [self.read_register(register + offset) for offset in range(length)]

    def set_registers(self, registers: Mapping[int, int]) -> None:
        for register, value in registers.items():
            self.registers[int(register) & 0xFF] = int(value) & 0xFF

    def written_registers(self) -> Dict[int, int]:
        """Return the last value written to each register."""

        return dict(self.write_log)

    def register_writes(self, register: int) -> List[int]:
        return [value for written, value in self.write_log if written == register]


class FakeSMBus:
    """Deterministic I2C bus that simulates a TCA9548A and its downstream devices.

    Devices are registered per channel (``None`` means the main bus, reachable
    only while every mux channel is disabled), which mirrors the real topology:
    three VL53L0X sensors can all sit at 0x29 because each is isolated on its
    own channel. Writes to the mux at ``mux_address`` are interpreted as channel
    select bytes exactly as the hardware does, including rejecting multi-channel
    masks that the drivers must never produce.
    """

    def __init__(
        self,
        mux_address: int = 0x70,
        devices: Optional[Mapping[Optional[int], Mapping[int, Mapping[int, int]]]] = None,
    ) -> None:
        self.mux_address = mux_address
        self._devices: Dict[Optional[int], Dict[int, FakeDevice]] = {}
        self._lock = threading.RLock()
        self.channel_mask: Optional[int] = None
        self.mux_writes: List[int] = []
        self.fail_mux_writes = 0
        self.faults: Dict[Tuple[Optional[int], int], Tuple[BaseException, Optional[int]]] = {}
        self.closed = False
        for channel, by_address in (devices or {}).items():
            for address, registers in by_address.items():
                self.add_device(channel, address, registers)

    # -- harness ---------------------------------------------------------

    def add_device(
        self, channel: Optional[int], address: int, registers: Optional[Mapping[int, int]] = None
    ) -> FakeDevice:
        if channel is not None and not 0 <= channel < TCA9548A_CHANNEL_COUNT:
            raise ValueError(f"invalid multiplexer channel {channel}")
        device = FakeDevice(address, registers)
        self._devices.setdefault(channel, {})[address] = device
        return device

    def device(self, channel: Optional[int], address: int) -> FakeDevice:
        try:
            return self._devices[channel][address]
        except KeyError as exc:
            where = "the main bus" if channel is None else f"channel {channel}"
            raise KeyError(f"no fake device at 0x{address:02X} on {where}") from exc

    def remove_device(self, channel: Optional[int], address: int) -> None:
        self._devices.get(channel, {}).pop(address, None)

    def inject_fault(
        self,
        channel: Optional[int],
        address: int,
        error: Optional[BaseException] = None,
        occurrences: Optional[int] = None,
    ) -> None:
        """Make the next ``occurrences`` accesses to a device raise (``None`` = forever)."""

        fault = error or OSError(errno.EREMOTEIO, f"injected fault at 0x{address:02X}")
        self.faults[(channel, address)] = (fault, occurrences)

    def clear_faults(self) -> None:
        self.faults.clear()

    @property
    def active_channel(self) -> Optional[int]:
        mask = self.channel_mask
        if not mask:
            return None
        return mask.bit_length() - 1

    # -- bus protocol ----------------------------------------------------

    def read_byte_data(self, address: int, register: int) -> int:
        self._check_open()
        with self._lock:
            if address == self.mux_address:
                if register != TCA9548A_CONTROL_REGISTER:
                    raise OSError(errno.EREMOTEIO, "TCA9548A exposes only control register 0")
                return self.channel_mask or 0
            device = self._resolve(address)
            return device.read_register(register)

    def write_byte_data(self, address: int, register: int, value: int) -> None:
        self._check_open()
        with self._lock:
            if address == self.mux_address:
                if register != TCA9548A_CONTROL_REGISTER:
                    raise OSError(errno.EREMOTEIO, "TCA9548A exposes only control register 0")
                self._select(value)
                return
            device = self._resolve(address)
            device.write_register(register, value)

    def read_i2c_block_data(self, address: int, register: int, length: int) -> List[int]:
        self._check_open()
        if length < 0:
            raise ValueError("block length must not be negative")
        with self._lock:
            if address == self.mux_address:
                raise OSError(errno.EREMOTEIO, "TCA9548A does not support block reads")
            device = self._resolve(address)
            return device.read_block(register, length)

    def write_byte(self, address: int, value: int) -> None:
        self._check_open()
        with self._lock:
            if address == self.mux_address:
                self._select(value)
                return
            device = self._resolve(address)
            device.command_log.append(value & 0xFF)

    def close(self) -> None:
        with self._lock:
            self.closed = True

    # -- internals -------------------------------------------------------

    def _check_open(self) -> None:
        if self.closed:
            raise ValueError("I2C bus is closed")

    def _select(self, mask: int) -> None:
        if self.fail_mux_writes > 0:
            self.fail_mux_writes -= 1
            raise OSError(errno.EREMOTEIO, "injected multiplexer write failure")
        if isinstance(mask, bool) or not isinstance(mask, int) or not 0 <= mask <= 0xFF:
            raise OSError(errno.EINVAL, f"invalid TCA9548A control byte {mask!r}")
        if bin(mask).count("1") > 1:
            raise OSError(errno.EINVAL, "TCA9548A must select at most one channel at a time")
        self.channel_mask = mask
        self.mux_writes.append(mask)

    def _resolve(self, address: int) -> FakeDevice:
        channel = self.active_channel
        fault = self.faults.get((channel, address))
        if fault is not None:
            error, occurrences = fault
            if occurrences is not None:
                if occurrences <= 1:
                    del self.faults[(channel, address)]
                else:
                    self.faults[(channel, address)] = (error, occurrences - 1)
            raise error
        device = self._devices.get(channel, {}).get(address)
        if device is None:
            where = "the main bus" if channel is None else f"channel {channel}"
            raise OSError(errno.EREMOTEIO, f"no device at 0x{address:02X} on {where}")
        return device


class TCA9548AMux:
    """Channel-selecting facade over one TCA9548A at 0x70 on I2C bus 1.

    All public methods acquire one re-entrant lock, so a channel selection and
    the transaction it enables can never interleave with another thread's. The
    selected channel is cached, and a channel is only rewritten when it really
    changes, because every redundant mux write costs a bus transaction in the
    30 Hz control loop. ``channel_context(None)`` writes ``0x00`` so nothing is
    connected to the main bus when it must be used directly.
    """

    def __init__(
        self,
        bus: SMBusProtocol,
        address: int = 0x70,
        retries: int = 2,
        retry_delay_s: float = 0.001,
    ) -> None:
        if not 0x70 <= address <= 0x77:
            raise ValueError("TCA9548A address must be in [0x70, 0x77]")
        if retries < 0:
            raise ValueError("retries must not be negative")
        self._bus = bus
        self.address = address
        self.retries = int(retries)
        self.retry_delay_s = max(0.0, float(retry_delay_s))
        self._lock = threading.RLock()
        self._selected_mask: Optional[int] = None
        self._scope_depth = 0
        self._closed = False

    @property
    def active_channel(self) -> Optional[int]:
        mask = self._selected_mask
        if not mask:
            return None
        return mask.bit_length() - 1

    @property
    def selected_mask(self) -> Optional[int]:
        return self._selected_mask

    def channel_context(self, channel: Optional[int]) -> ContextManager[SMBusProtocol]:
        self._validate_channel(channel)
        return _ChannelScope(self, channel)

    def read_byte(self, channel: Optional[int], address: int, register: int) -> int:
        with self.channel_context(channel) as bus:
            return int(bus.read_byte_data(address, register)) & 0xFF

    def write_byte(self, channel: Optional[int], address: int, register: int, value: int) -> None:
        if not 0 <= value <= 0xFF:
            raise ValueError("register value must fit in one byte")
        with self.channel_context(channel) as bus:
            bus.write_byte_data(address, register, value)

    def read_block(
        self, channel: Optional[int], address: int, register: int, length: int
    ) -> List[int]:
        if length <= 0:
            raise ValueError("block length must be positive")
        with self.channel_context(channel) as bus:
            values = bus.read_i2c_block_data(address, register, length)
        return [int(value) & 0xFF for value in values]

    def scan_channel(self, channel: int) -> List[int]:
        """Probe every 7-bit address on one channel and return those that answer."""

        self._validate_channel(channel)
        found: List[int] = []
        with self.channel_context(channel) as bus:
            for address in range(0x08, 0x78):
                if address == self.address:
                    continue
                try:
                    bus.read_byte_data(address, 0x00)
                except OSError:
                    continue
                found.append(address)
        return found

    def disable_all(self) -> None:
        with self._lock:
            self._ensure_open()
            self._select_locked(None)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._select_locked(None, force=True)
            finally:
                self._closed = True
                self._selected_mask = None
                self._bus.close()

    # -- internals -------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("TCA9548A multiplexer is closed")

    @staticmethod
    def _validate_channel(channel: Optional[int]) -> None:
        if channel is None:
            return
        if isinstance(channel, bool) or not isinstance(channel, int):
            raise ValueError("multiplexer channel must be an integer or None")
        if not 0 <= channel < TCA9548A_CHANNEL_COUNT:
            raise ValueError(f"multiplexer channel must be in [0, {TCA9548A_CHANNEL_COUNT - 1}]")

    def _select_locked(self, channel: Optional[int], force: bool = False) -> None:
        mask = 0 if channel is None else 1 << channel
        if not force and self._selected_mask == mask:
            return
        attempts = 0
        while True:
            try:
                self._bus.write_byte(self.address, mask)
            except OSError:
                # The mux state is unknown after a failed write, so the cache is
                # dropped and the next transaction reselects unconditionally.
                self._selected_mask = None
                if attempts >= self.retries:
                    raise
                attempts += 1
                if self.retry_delay_s:
                    time.sleep(self.retry_delay_s)
                continue
            self._selected_mask = mask
            return


class _ChannelScope:
    """Context manager returned by :meth:`TCA9548AMux.channel_context`."""

    __slots__ = ("_mux", "_channel", "_entered", "_previous_mask", "_previous_known")

    def __init__(self, mux: TCA9548AMux, channel: Optional[int]) -> None:
        self._mux = mux
        self._channel = channel
        self._entered = False
        self._previous_mask: Optional[int] = None
        self._previous_known = False

    def __enter__(self) -> SMBusProtocol:
        TCA9548AMux._validate_channel(self._channel)
        self._mux._lock.acquire()
        try:
            self._mux._ensure_open()
            self._previous_mask = self._mux._selected_mask
            self._previous_known = self._previous_mask is not None
            self._mux._select_locked(self._channel)
            self._mux._scope_depth += 1
            self._entered = True
        except BaseException:
            self._mux._lock.release()
            raise
        return self._mux._bus

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if not self._entered:
            return None
        try:
            self._mux._scope_depth -= 1
            # A nested scope must not leave the enclosing block talking to the
            # wrong device; a top-level scope leaves the mux where it is so the
            # next transaction on the same channel can skip the select write.
            if self._mux._scope_depth > 0 and self._previous_known:
                previous_mask = self._previous_mask or 0
                restore_channel = (
                    None if not previous_mask else previous_mask.bit_length() - 1
                )
                if self._mux._selected_mask != previous_mask:
                    try:
                        self._mux._select_locked(restore_channel)
                    except OSError:
                        pass
        finally:
            self._mux._lock.release()
        return None
