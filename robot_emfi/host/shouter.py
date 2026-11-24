#!/usr/bin/env python3
"""ChipSHOUTER fired by the Husky's glitch output (MCX) on its SMB hardware trigger.

The SMB input drives the output switch directly, past the firmware: the EM pulse lasts as long
as the trigger pulse is high, and pulse.width/repeat/deadtime only apply to the front-panel
button, the pulse command and the RJ12 pin. So the Husky's glitch length (scope.glitch.repeat
cycles of the 50 MHz clock, 20 ns each) IS the pulse width here; keep it within the
ChipSHOUTER's 80-960 ns. The firmware still watches the input: a trigger while disarmed latches
trig_glitch, one held high > 10 ms a trigger fault.

It cannot report whether it fired. 'armed' before and after a Husky trigger is the best it
gives; the pulse itself shows in the Husky trace (peak at the glitch time).

Keep it disarmed whenever the MCX is not the configured short glitch: setup_glitch() puts the
scope trigger (milliseconds long) on the MCX while it reconfigures the glitch module, and with
the glitch module's default output (clock_xor) the MCX carries the 50 MHz clock, which the
ChipSHOUTER sees as a trigger held high (fault_trigger_error, tested 2026-10-30). Low (inert):
output enable_only, whether the glitch module is enabled or not.

usage (status only, never arms): shouter.py
"""
import time

from chipshouter import ChipSHOUTER
from chipshouter.com_tools import Firmware_State_Exception, Max_Retry_Exception, Reset_Exception

from site_config import SHOUTER_PORT as PORT
WIDTH_NS = (80, 960)
VOLTS = (150, 500)
ARMING_S = 15                               # arm again if it is still 'arming' after this long
ERRORS = (Firmware_State_Exception, Max_Retry_Exception, IOError, OSError)


class Shouter:
    def __init__(self, voltage=None, width_ns=None, term_50=True, port=PORT):
        """voltage None: keep its setting, only to hold it disarmed (it cannot be armed then)."""
        assert voltage is None or VOLTS[0] <= voltage <= VOLTS[1], f"ChipSHOUTER voltage {voltage} outside {VOLTS} V"
        self.voltage = None if voltage is None else int(voltage)
        self.width_ns, self.term_50, self.port = width_ns, term_50, port
        self.resets = 0
        self.t_arm = 0.0                        # when it was last told to arm
        self.last = ""                          # last problem, for the console
        self.cs = ChipSHOUTER(port)
        self.cs.armed = False
        self.configure()

    def configure(self):
        cs = self.cs
        if self.voltage is not None:
            cs.voltage = self.voltage
        if self.width_ns is not None:          # same width for its own generator (80 ns at least), in case
            cs.pulse.width = max(WIDTH_NS[0], self.width_ns)   # the SMB trigger only starts the pulse
        cs.hwtrig_mode = True                   # active high: the Husky glitch output idles low
        cs.hwtrig_term = self.term_50           # 50 ohm: clean edges on the coax; False = ~1.8 kohm
        cs.faults_current = 0                   # clear stale faults (e.g. trig_glitch while disarmed)
        assert self.voltage in (None, cs.voltage.set) and cs.hwtrig_mode, "ChipSHOUTER did not take the settings"

    def _rebooted(self):
        """The ChipSHOUTER reset (EMFI can do that to it). It comes back disarmed, maybe with its
        stored settings: wait as the API asks, then configure it again."""
        self.resets += 1
        time.sleep(5)
        try:
            self.cs.com_api.ready_for_commands()
        except (Reset_Exception, Max_Retry_Exception):
            self.cs = ChipSHOUTER(self.port)
        self.cs.armed = False
        self.configure()

    def ready(self, timeout=0.4):
        """Armed, so that the next Husky trigger fires it? Arms it if not (clearing faults first,
        and checking the voltage setting survived). False if that takes longer than timeout:
        the HV charges for up to a few seconds after arming; the caller skips this shot.
        States seen: disarmed, arming, armed, fault_disarmed. While it is arming it is left alone
        (arming again restarts the charge), unless that has taken over ARMING_S."""
        assert self.voltage is not None, "no voltage given: this Shouter only holds it disarmed"
        try:
            st = self.cs.state
            if st == "armed":
                return True
            if st != "arming" or time.monotonic() - self.t_arm > ARMING_S:
                if "fault" in st:
                    self.last = f"{st} {self.cs.faults_latched}"
                if self.cs.voltage.set != self.voltage or not self.cs.hwtrig_mode:
                    self.configure()
                self.cs.clr_armed = True
                self.t_arm = time.monotonic()
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                st = self.cs.state
                if st == "armed":
                    return True
            self.last = f"{st} {time.monotonic() - self.t_arm:.1f} s after arming"
        except Reset_Exception:
            self.last = "reset"
            self._rebooted()
        except ERRORS as exc:
            self.last = f"{type(exc).__name__}: {exc}"
        return False

    def after_shot(self):
        """(state, latched faults) after a shot: 'armed' with no faults is all it can confirm."""
        try:
            st = self.cs.state
            return st, ([] if st == "armed" else self.cs.faults_latched)
        except Reset_Exception:
            self._rebooted()
            return "reset", []
        except ERRORS as exc:
            return f"{type(exc).__name__}", []

    def temps(self):
        """(mosfet, transformer, diode) in C, None on a read error. The transformer sensor has read
        0 since 2026-10-30 ~17:40, and all three froze at 59 once before a reboot."""
        try:
            return self.cs.temperature_mosfet, self.cs.temperature_xformer, self.cs.temperature_diode
        except Reset_Exception:
            self._rebooted()
        except ERRORS:
            pass
        return None

    def measured_v(self):
        """Capacitor bank voltage now (V; ~20 V disarmed), None on a read error."""
        try:
            return self.cs.voltage.measured
        except Reset_Exception:
            self._rebooted()
        except ERRORS:
            pass
        return None

    def disarm(self):
        """True once it reports disarmed (a reset also leaves it disarmed)."""
        for _ in range(3):
            try:
                self.cs.armed = False
                if "disarmed" in self.cs.state:   # not 'arming' either: that charges the HV
                    return True
            except Reset_Exception:
                time.sleep(5)
            except ERRORS as exc:
                self.last = f"disarm: {type(exc).__name__}: {exc}"
                time.sleep(0.5)
        return False

    def close(self):
        ok = self.disarm()
        try:
            self.cs.disconnect()
        except ERRORS:
            pass
        return ok


if __name__ == "__main__":
    cs = ChipSHOUTER(PORT)
    print(f"{cs.id.strip()}: {cs.state}, faults {cs.faults_current} latched {cs.faults_latched}")
    print(f"  voltage set {cs.voltage.set} V, measured {cs.voltage.measured} V; hwtrig "
          f"{'active-high' if cs.hwtrig_mode else 'active-low'}, {'50 ohm' if cs.hwtrig_term else 'high-Z'}; "
          f"arm timeout {cs.arm_timeout} min")
    print(f"  temperatures mosfet {cs.temperature_mosfet}, transformer {cs.temperature_xformer}, "
          f"diode {cs.temperature_diode} C")
