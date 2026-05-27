---
standard: MIL-STD-883
method: "2020.8"
category: general
language: en
---

# MIL-STD-883 Method 2020.8

* METHOD 2020.8
PARTICLE IMPACT NOISE DETECTION TEST
1. PURPOSE. The purpose of this test is to detect loose particles inside a device cavity. The test provides a
nondestructive means of identifying those devices containing particles of sufficient mass that, upon impact with the case,
excite the transducer.
2. APPARATUS. The equipment required for the particle impact noise detection (PIND) test shall consist of the following
(or equivalent):
a. A threshold detector to detect particle noise voltage exceeding a preset threshold of the absolute value of 20 ±1
millivolt peak reference to system ground.
b. A vibration shaker and driver assembly capable of providing essentially sinusoidal motion to the device under test
(DUT) at:
(1) Condition A: 20 g peak at 40 to 250 Hz.
(2) Condition B: 10 g peak at 60 Hz minimum.
c. PIND transducer, calibrated to a peak sensitivity of -77.5 ±3 dB in regards to one volt per microbar at a point within
the frequency of 150 to 160 kHz.
d. A sensitivity test unit (STU) (see figure 2020-1) for periodic assessment of the PIND system performance. The
STU shall consist of a transducer with the same tolerances as the PIND transducer and a circuit to excite the
transducer with a 250 microvolt ±20 percent pulse. The STU shall produce a pulse of about 20 mV peak on the
oscilloscope when the transducer is coupled to the PIND transducer with attachment medium.
e. PIND electronics, consisting of an amplifier with a gain of 60 ±2 dB centered at the frequency of peak sensitivity of
the PIND transducer. The noise at the output of the amplifier shall not exceed 10 mV peak.
f. Attachment medium. The attachment medium used to attach the DUT to the PIND transducer shall be the same
attachment medium as used for the STU test.
g. Shock mechanism or tool capable of imparting shock pulses of 1,000 ±200 g peak to the DUT. The duration of the
main shock shall not exceed 100 µs. If an integral co-test shock system is used the shaker vibration may be
interrupted or perturbed for period of time not to exceed 250 ms from initiation of the last shock pulse in the
sequence. The co-test duration shall be measured at the 50 ±5 percent point.
3. PROCEDURES.
3.1 Test equipment setup. Shaker drive frequency and amplitude shall be adjusted to the specified conditions based on
cavity size of the DUT (for condition A, see table 1 herein). The shock pulse shall be adjusted to provide 1,000 ±200 g peak
to the DUT.
3.2 Test equipment checkout. The test equipment checkout shall be performed a minimum of one time per operation
shift. Failure of the system to meet checkout requirements shall require retest of all devices tested subsequent to the last
successful system checkout.
3.2.1 Shaker drive system checkout. The drive system shall achieve the shaker frequency and the shaker amplitude
specified. The drive system shall be calibrated so that the frequency settings are within ±8 percent and the amplitude
vibration setting are within ±10 percent of the nominal values. If a visual displacement monitor is affixed to the transducer, it
may be used for amplitudes between 0.04 and 0.12 inch (1.02 and 3.05 mm). An accelerometer may be used over the
entire range of amplitudes and shall be used below amplitudes of 0.040 inch (1.02 mm).
METHOD 2020.8
18 June 2004
1

3.2.2 Detection system checkout. With the shaker deenergized, the STU transducer shall be mounted face-to-face and
coaxial with the PIND transducer using the attachment medium used for testing the devices. The STU shall be activated
several times to verify low level signal pulse visual and threshold detection on the oscilloscope. Not every application of the
STU will produce the required amplitude. All pulses which are greater than 20 mV shall activate the detector.
3.2.3 System noise verification. System noise will appear as a fairly constant band and must not exceed 20 millivolts
peak to peak when observed for a period of 30 to 60 seconds.
3.3 Test sequence. The following sequence of operations (a through i) constitute one test cycle or run.
a. 3 pre-test shocks.
b. Vibration 3 ±1 seconds.
c. 3 co-test shocks.
d. Vibration 3 ±1 seconds.
e. 3 co-test shocks.
f. Vibration 3 ±1 seconds.
g. 3 co-test shocks.
h. Vibration 3 ±1 seconds.
i. Accept or reject.
3.3.1 Mounting requirements. Special precautions (e.g., in mounting, grounding of DUT leads, or grounding of test
operator) shall be taken as necessary to prevent electrostatic damage to the DUT.
* Batch or bulk testing is prohibited.
* Most part types will mount directly to the transducer via the attachment medium. Parts shall be mounted with the
largest flat surface against the transducer at the center or axis of the transducer for maximum sensitivity. The DUT
shall be placed such that the geometric center of the surface contacting the transducer is centrally located on the
transducer to within approximately 2 mm of the transducer surface’s geometric center. Where more than one large
surface exists, the one that is the thinnest in section or has the most uniform thickness shall be mounted toward the
transducer, e.g., flat packs are mounted top down against the transducer. Small axial-lead, right circular cylindrical
parts are mounted with their axis horizontal and the side of the cylinder against the transducer. Parts with unusual
shapes may require special fixtures. Such fixtures shall have the following properties:
(1) Low mass.
(2) High acoustic transmission (aluminum alloy 7075 works well).
(3) Full transducer surface contact, especially at the center.
(4) Maximum practical surface contact with test part.
(5) No moving parts.
(6) Suitable for attachment medium mounting.
METHOD 2020.8
18 June 2004
2

| * |
| --- |
| * |

3.3.2 Test monitoring. Each test cycle (see 3.3) shall be continuously monitored, except for the period during co-test
shocks and 250 ms maximum after the shocks. Particle indications can occur in any one or combinations of the three
detection systems as follows:
a. Visual indication of high frequency spikes which exceed the normal constant background white noise level.
b. Audio indication of clicks, pops, or rattling which is different from the constant background noise present with no
DUT on the transducer.
c. Threshold detection shall be indicated by the lighting of a lamp or by deflection of the secondary oscilloscope trace.
3.4 Failure criteria. Any noise bursts as detected by any of the three detection systems exclusive of background noise,
except those caused by the shock blows, during the monitoring periods shall be cause for rejection of the device. Rejects
shall not be retested except for retest of all devices in the event of test system failure. If additional cycles of testing on a lot
are specified, the entire test procedure (equipment setup and checkout mounting, vibration, and co-shocking) shall be
repeated for each retest cycle. Reject devices from each test cycle shall be removed from the lot and shall not be retested
in subsequent lot testing.
3.5 Screening lot acceptance. Unless otherwise specified, the inspection lot (or sublot) to be screened for lot acceptance
*
shall be submitted to 100 percent PIND testing a maximum of five times in accordance with condition A herein. PIND
prescreening shall not be performed. The lot may be accepted on any of the five runs if the percentage of defective devices
in that run is less than 1 percent and the cumulative number of defective devices does not exceed 25 percent. All defective
devices shall be removed after each run. Resubmission is not allowed.
METHOD 2020.8
18 June 2004
3

* TABLE I. Package Height vs.Test Frequency for 20g Acceleration (condition A).
Note: The shaker drive test frequency (F) for condition A (see 3.1) is determined by the package internal cavity height
using the following formula:
F = 20 /[( D)X (0.0511 )]
where: D = Average internal package height (in inches).
20 is a constant in this application and is equal to sinusoidal acceleration of 20g.
F is the shaker drive test frequency (in Hz)
Based on the formula above, the following table is generated:
Average Internal Cavity Height Test Frequency
Mils mm inches Hz
30 0.76 0.030 114
40 1.02 0.040 99
50 1.27 0.050 88
60 1.52 0.060 81
70 1.78 0.070 75
80 2.13 0.080 70
90 2.29 0.090 66
100 2.54 0.100 63
110 2.79 0.110 60
Example calculation: Assume an average internal cavity height of 70 Mils.
F = 20 /[( D)X (0.0511 )]
D = 70 Mils converted to inches = .070 inches.
F = 20 /[(. 070 )X (0.0511 )] = 20 /[.00358 ] = 5586 = 75 Hz
4. SUMMARY. The following details shall be specified in the applicable acquisition document:
a. Test condition letter A or B.
b. Lot acceptance/rejection criteria (if other than specified in 3.5).
c. The number of test cycles, if other than one.
d. Pre-test shock level and co-test shock level, if other than specified.
METHOD 2020.8
18 June 2004
4

| Average Internal Cavity Height |  |  | Test Frequency |
| --- | --- | --- | --- |
| Mils | mm | inches | Hz |
| 30 40 50 60 70 80 90 100 110 | 0.76 1.02 1.27 1.52 1.78 2.13 2.29 2.54 2.79 | 0.030 0.040 0.050 0.060 0.070 0.080 0.090 0.100 0.110 | 114 99 88 81 75 70 66 63 60 |

NOTES:
1. Pushbutton switch: Mechanically quiet, fast make, gold contacts. E.G. T2 SM4 microswitch.
2. Resistance tolerance 5 percent noninductive.
3. Voltage source can be a standard dry cell.
4. The coupled transducers must be coaxial during test.
5. Voltage output to STU transducer 250 microvolts, ±20 percent.
FIGURE 2020-1 Typical sensitivity test unit.
METHOD 2020.8
18 June 2004
5

METHOD 2020.8
18 June 2004
6
