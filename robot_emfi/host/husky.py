#!/usr/bin/env python3
"""ChipWhisperer-Husky set-up that the campaigns share: the probe input, the glitch output that
triggers the ChipSHOUTER, and the delay lists of the sweeps. Imported, not run.
"""
import time

CLK = 50e6                 # Husky clock: one glitch cycle = 20 ns


def setup(scope, gain_db):
    """Probe input on MEASURE at CLK, glitch module off and inert."""
    scope.clock.clkgen_src = "system"
    scope.clock.clkgen_freq = CLK
    scope.clock.adc_mul = 1
    scope.glitch.enabled = False
    scope.glitch.output = "enable_only"   # the default (clock_xor) puts the clock on the MCX even when disabled
    scope.io.glitch_hp = False
    scope.io.glitch_lp = False
    scope.io.glitch_trig_mcx = "glitch"
    scope.gain.db = gain_db
    scope.adc.timeout = 1.0
    time.sleep(0.2)
    assert scope.clock.adc_locked, "ADC clock not locked"


def setup_glitch(scope, level, pulse_cycles):
    """One glitch of pulse_cycles cycles on the MCX output, ext_offset cycles after the ADC trigger."""
    # Keep the MCX off the glitch path while the glitch module is enabled and reconfigured: its
    # default output mode puts a clock on the MCX, which the injector would see as triggers. MCX last.
    scope.io.glitch_trig_mcx = "trigger"
    scope.trigger.module = "ADC"
    scope.trigger.level = level
    scope.glitch.enabled = True
    scope.glitch.clk_src = "pll"
    scope.glitch.output = "enable_only"
    scope.glitch.trigger_src = "ext_single"
    scope.glitch.num_glitches = 1
    scope.glitch.ext_offset = 0
    scope.glitch.repeat = pulse_cycles
    time.sleep(0.1)
    assert scope.glitch.mmcm_locked, "Husky glitch MMCM not locked"
    scope.adc.decimate = 1
    scope.adc.presamples = 0
    scope.adc.samples = 1000
    scope.adc.timeout = 0.6
    scope.io.glitch_trig_mcx = "glitch"


def parse_delays(spec):
    """'a:b:step' (b excluded) or 'a,b,c', in ms."""
    if ":" in spec:
        a, b, s = (float(x) for x in spec.split(":"))
        out, d = [], a
        while d < b:
            out.append(round(d, 3))
            d += s
        return out
    return [float(x) for x in spec.split(",")]
