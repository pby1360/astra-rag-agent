---
standard: MIL-STD-883
method: "1005.8"
category: general
language: en
---

# MIL-STD-883 Method 1005.8 — STEADY-STATE LIFE

METHOD 1005.8
STEADY-STATE LIFE
1. PURPOSE. The steady-state life test is performed for the purpose of demonstrating the quality or reliability of devices
subjected to the specified conditions over an extended time period. Life tests conducted within rated operating conditions
should be conducted for a sufficiently long test period to assure that results are not characteristic of early failures or "infant
mortality," and periodic observations of results should be made prior to the end of the life test to provide an indication of any
significant variation of failure rate with time. Valid results at shorter intervals or at lower stresses require accelerated test
conditions or a sufficiently large sample size to provide a reasonable probability of detection of failures in the sample
corresponding to the distribution of potential failures in the lot(s) from which the sample was drawn. The test conditions
provided in 3 below are intended to reflect these considerations.
When this test is employed for the purpose of assessing the general capability of a device or for device qualification tests in
support of future device applications requiring high reliability, the test conditions should be selected so as to represent the
maximum operating or testing (see test condition F) ratings of the device in terms of electrical input(s), load and bias and the
corresponding maximum operating or testing temperature or other specified environment.
2. APPARATUS. Suitable sockets or other mounting means shall be provided to make firm electrical contact to the
terminals of devices under test in the specified circuit configuration. Except as authorized by the acquiring or qualifying
activity, the mounting means shall be so designated that they will not remove internally-dissipated heat from the device by
conduction, other than that removed through the device terminals, the necessary electrical contacts and the gas or liquid
chamber medium. The apparatus shall provide for maintaining the specified biases at the terminals of the device under test
and, when specified, monitoring of the input excitation or output response. Power supplies and current-setting resistors shall
be capable of maintaining the specified operating conditions as minimal throughout the testing period, despite normal
variations in source voltages, ambient temperatures, etc. When test conditions result in significant power dissipation, the
test apparatus shall be arranged so as to result in the approximate average power dissipation for each device whether
devices are tested individually or in a group. The test circuits need not compensate for normal variations in individual device
characteristics, but shall be so arranged that the existence of failed or abnormal (i.e., open, short, etc.) devices in a group
does not negate the effect of the test for other devices in the group.
3. PROCEDURE. The microelectronic devices shall be subjected to the specified test condition (see 3.5) for the
specified duration at the specified test temperature, and the required measurements shall be made at the specified
intermediate points and end points. QML manufactures who are certified and qualified to MIL-PRF-38535 may modify the
time or the condition independently from the regression conditions contained in table I or the test condition/circuit specified
in the device specification or standard microcircuit drawing provided the modification is contained in the manufacturer's QM
plan and the "Q" certification identifier is marked on the devices. Lead-, stud-, or case-mounted devices shall be mounted
by the leads, stud, or case in their normal mounting configuration, and the point of connection shall be maintained at a
temperature not less than the specified ambient temperature. The test condition, duration, sample size, and temperature
selected prior to test shall be recorded and shall govern for the entire test. Test boards shall not employ load resistors
which are common to more than one device, or to more than one output pin on the same device.
3.1 Test duration.
3.1.1 Test duration - standard life. The life test duration shall be 1,000 hours minimum at 125°C , unless otherwise
specified or allowed (see 3.2.1). After the specified duration of the test, the device shall be removed from the test conditions
and allowed to reach standard test conditions. Where the purpose of this test is to demonstrate compliance with a specified
lambda (8), the test may be terminated at the specified duration or at the point of rejection if this occurs prior to the specified
test duration.
METHOD 1005.8
1 June 1993
1

3.1.2 Accelerated life test duration. For class level B, the life test duration, when accelerated, shall be the time equivalent
to 1,000 hours at 125°C for the ambient temperature selected or specified (see table I). Within 72 hours after the specified
duration of the test, the device shall be removed from the specified test conditions and allowed to reach standard test
conditions without removal of bias. The interruption of bias for up to one minute for the purpose of moving the devices to
cool-down positions separate from the chamber within which life testing was performed shall not be considered removal of
bias.
3.2 Test temperature. The specified test temperature is the minimum ambient temperature to which all devices in the
working area of the chamber shall be exposed. This shall be assured by making whatever adjustments are necessary in the
chamber profile, loading, location of control or monitoring instruments, and the flow of air or other suitable gas or liquid
chamber medium. Therefore, calibration shall be accomplished in the chamber in a fully loaded, (boards need not be loaded
with devices) unpowered configuration, and the indicator sensor located at, or adjusted to reflect, the coldest point in the
working area.
3.2.1 Test temperature - standard life. Unless otherwise specified, the ambient life test temperature shall be 125°C
minimum for test conditions A through E (see 3.5), except that for hybrid microcircuits, the conditions may be modified in
accordance with table I. At the supplier's option, the ambient temperature for conditions A through E may be increased and
the test duration reduced in accordance with table I using the specified test circuit and bias conditions. Since case and
junction temperature will, under normal circumstances, be significantly higher than ambient temperature, the circuit should
be so structured that maximum rated case or junction temperatures for test or operation shall not exceed 200°C for class
level B or 175°C for class level S (see 3.2.1.1).
3.2.1.1 Test temperature for high power devices. Regardless of power level, devices shall be able to be burned in or
life-tested at their maximum rated operating temperature. For devices whose maximum operating temperature is stated in
terms of ambient temperature, T , table I applies. For devices whose maximum operating temperature is stated in terms of
A
case temperature, T , and where the ambient temperature would cause T to exceed +200°C (+175°C for class level S),
C J
the ambient operating temperature may be reduced during burn-in and life test from +125°C to a value that will demonstrate
a T between +175°C and +200°C and T equal to or greater than +125°C without changing the test duration. Data
J C
supporting this reduction shall be available to the acquiring and qualifying activities upon request.
3.2.1.2 Test temperature for hybrid devices. The ambient or case life test temperature shall be as specified in table I,
except case temperature life test shall be performed, as a minimum, at the maximum operating case temperature (T )
C
specified for the device. Life test shall be for 1,000 hours minimum for class level S hybrid (class K). The device should be
life tested at the maximum specified operating temperature, voltage, and loading conditions as specified in the detail
specification. Since case and junction temperature will, under normal circumstances, be significantly higher than ambient
temperature, the circuit should be so structured that the maximum rated junction temperature as specified in the device
specification or drawing and the cure temperature of polymeric materials as specified in the baseline documentation shall
not be exceeded. If no maximum junction temperature is specified, a maximum of 175°C is assumed. Accelerated life test
(condition F) shall not be permitted. The specified test temperature shall be the minimum actual ambient or case
temperature that must be maintained for all devices in the chamber. This shall be assured by making whatever adjustments
are necessary in the chamber profile, loading, location of control or monitoring instruments and the flow of air or other
suitable gas or liquid chamber medium.
3.2.2 Test temperature - accelerated life. When condition F is specified or is utilized as an option (when allowed by the
applicable acquisition documents), the minimum ambient test temperature shall be +175°C , unless otherwise specified.
Since accelerated testing will normally be performed at temperatures higher than the maximum rated operating junction
temperature of the device(s) tested, care shall be taken to ensure that the device(s) does not go into thermal runaway.
3.2.3 Special considerations for devices with internal thermal limitation using test conditions A through E. For devices
with internal thermal shutdown, extended exposure at a temperature in excess of the shut-down temperature will not provide
a realistic indicator of long-term operating reliability. For devices equipped with thermal shutdown, operating life test shall be
performed at an ambient temperature where the worst case junction temperature is at least 5°C below the worst case
thermal shutdown threshold. Data supporting the defined thermal shutdown threshold shall be available to the preparing or
acquiring activity upon request.
METHOD 1005.8
1 June 1993
2

3.3 Measurements.
3.3.1 Measurements for test temperatures less than or equal to 150°C . Unless otherwise specified, all specified
intermediate and end-point measurements shall be completed within 96 hours after removal of the device from the specified
test conditions (i.e., either removal of temperature or bias). If these measurements cannot be completed within 96 hours,
the devices shall be subjected to the same test condition (see 3.5) and temperature previously used for a minimum of 24
additional hours before intermediate or end-point measurements are made. When specified (or at the manufacturer's
discretion, if not specified), intermediate measurements shall be made at 168 (+72, -0) hours and at 504 (+168, -0) hours.
For tests in excess of 1,000 hours duration, additional intermediate measurement points, when specified, shall be 1000
(+168, -24) hours, 2,000 (+168, -24) hours, and each succeeding 1,000 (+168, -24) hour interval. These intermediate
measurements shall consist of the parameters and conditions specified, including major functional characteristics of the
device under test, sufficient to reveal both catastrophic and degradation failures to specified limits. Devices shall be cooled
to less than 10°C of their power stable condition at room temperature prior to the removal of bias.
The interruption of bias for up to one minute for the purpose of moving the devices to cool-down positions separate from the
chamber within which life testing was performed shall not be considered removal of bias. Alternatively, except for linear or
MOS (CMOS, NMOS, PMOS, etc.) devices or unless otherwise specified, the bias may be removed during cooling, provided
the case temperature of the devices under test is reduced to a maximum of 35°C within 30 minutes after removal of the test
conditions and provided the devices under test are removed from the heated chamber within five minutes following removal
of bias. All specified 25°C electrical measurements shall be completed prior to any reheating of the device(s).
3.3.2 Measurements for test temperatures greater than or equal to 175°C . Unless otherwise specified, all specified
intermediate and end-point measurements shall be completed within 24 hours after removal of the device from the specified
test conditions (i.e., either removal of temperature or bias). If these measurements cannot be completed within 24 hours,
the steady-state life test shall be repeated using the same test condition, temperature and time. Devices shall be cooled to
less than 10°C of their power stable condition at room temperature prior to the removal of bias, except that the interruption
of bias for up to one minute for the purpose of moving the devices to cool-down positions shall not be considered removal of
bias. All specified 25°C electrical measurements shall be completed prior to any reheating of the device(s).
3.3.3 Test setup monitoring. The test setup shall be monitored at the test temperature initially and at the conclusion of
the test to establish that all devices are being stressed to the specified requirements. The following is the minimum
acceptable monitoring procedure:
a. Device sockets. Initially and at least each 6 months thereafter, (once every 6 months or just prior to use if not
used during the 6 month period) each test board or tray shall be checked to verify continuity to connector points
to assure that bias supplies and signal information will be applied to each socket. Board capacitance or
resistance required to ensure stability of devices under test shall be checked during these initial and periodic
verification tests to ensure they will perform their proper function (i.e., that they are not open or shorted). Except
for this initial and periodic verification, each device or device socket does not have to be checked; however,
random sampling techniques shall be applied prior to each time a board is used and shall be adequate to assure
that there are correct and continuous electrical connections to the devices under test.
b. Connectors to test boards or trays. After the test boards are loaded with devices, inserted into the oven, and
brought up to at least 125°C or the specified test temperature, whichever is less, each required test voltage and
signal condition shall be verified in at least one location on each test board or tray so as to assure electrical
continuity and the correct application of specified electrical stresses for each connection or contact pair used in
the applicable test configuration. This may be performed by opening the oven for a maximum of 10 minutes.
When the test conditions are checked at a test socket, contact points on the instrument used to make this
continuity check shall be equal to or smaller dimensions than the leads (contacts) of the devices to be tested and
shall be constructed such that the socket contacts are not disfigured or damaged.
c. At the conclusion of the test period, prior to removal of devices from temperature and test conditions, the voltage
and signal condition verification of b. above shall be repeated.
d. For class level S devices, each test board or tray and each test socket shall be verified prior to test to assure that
the specified test conditions are applied to each device. This may be accomplished by verifying the device
functional response at each device output(s). An approved alternate procedure may be used.
Where failures or open contacts occur which result in removal of the required test stresses for any period of the required test
duration (see 3.1), the test time shall be extended to assure actual exposure for the total minimum specified test duration.
Any loss(es) or interruption(s) of bias in excess of 10 minutes total duration whether or not the chamber is at temperature
during the final 24 hours of life test shall require extension of the test duration for an uninterrupted 24 hours minimum, after
the last bias interruption.
METHOD 1005.8
1 June 1993
3

3.4 Test sample. The test sample shall be as specified (see 4). When this test method is employed as an add-on life
test for a series or family of device types, lesser quantities of any single device type may be introduced in any single addition
to the total sample quantity, but the results shall not be considered valid until the minimum sample size for each device has
been accumulated. Where all or part of the samples previously under test are extracted upon addition of new samples, the
minimum sample size for each type shall be maintained once that level is initially reached and no sample shall be extracted
until it has accumulated the specified minimum test hours (see 3.1).
3.5 Test conditions.
3.5.1 Test condition A, steady-state, reverse bias. This condition is illustrated on figure 1005-1 and is suitable for use on
all types of circuits, both linear and digital. In this test, as many junctions as possible will be reverse biased to the specified
voltage.
3.5.2 Test condition B, steady-state, forward bias. This test condition is illustrated on figure 1005-1 and can be used on
all digital type circuits and some linear types. In this test, as many junctions as possible will be forward biased as specified.
3.5.3 Test condition C, steady-state, power and reverse bias. This condition is illustrated on figure 1005-1 and can be
used on all digital type circuits and some linear types where the inputs can be reverse biased and the output can be biased
for maximum power dissipation or vice versa.
3.5.4 Test condition D, parallel excitation. This test condition is typically illustrated on figure 1005-2 and is suitable for
use on all circuit types. All circuits must be driven with an appropriate signal to simulate, as closely as possible, circuit
application and all circuits shall have maximum load applied. The excitation frequency shall not be less than 60 Hz.
3.5.5 Test condition E, ring oscillator. This test condition is illustrated on figure 1005-3, with the output of the last circuit
normally connected to the input of the first circuit. The series will be free running at a frequency established by the
propagation delay of each circuit and associated wiring and the frequency shall not be less than 60 Hz. In the case of
circuits which cause phase inversion, an odd number of circuits shall be used. Each circuit in the ring shall be loaded to its
rated maximum. While this condition affords the opportunity to continuously monitor the test for catastrophic failures (i.e.,
ring stoppage), this shall not be considered acceptable as a substitute for the intermediate measurements (see 3.3).
3.5.6 Test condition F, (class level B only) temperature-accelerated test. In this test condition, microcircuits are
subjected to bias(es) at an ambient test temperature (175°C to 300°C ) which considerably exceeds their maximum rated
temperature. At higher temperatures, it is generally found that microcircuits will not operate normally, and it is therefore
necessary that special attention be given to the choice of bias circuits and conditions to assure that important circuit areas
are adequately biased without damaging overstresses to other areas of the circuit. To properly select the actual biasing
conditions to be used, it is recommended that an adequate sample of devices be exposed to the intended high temperature
while measuring voltage(s) and current(s) at each device terminal to assure that the specified circuit and the applied
electrical stresses do not induce damaging overstresses.
At the manufacturer's option, alternate time and temperature values may be established from table I. Any time-temperature
combination which is contained in table I within the time limit of 30 to 1,000 hours may be used. The life test ground rules of
3.5 of method 1016 shall apply to life tests conducted using test condition F. The applied voltage at any or all terminals shall
be equal to the voltage specified for the 125°C operating life in the applicable acquisition document, unless otherwise
specified.
If necessary, with the specific approval of the qualifying activity, the applied voltage at any or all terminal(s) may be reduced
to not less than 50 percent of the specified value(s) when it is demonstrated that excessive current flow or power dissipation
would result from operation at the specified voltage(s). If the voltage(s) is so reduced, the life test duration shall be
determined by the following formula:
t (100%)
o
T =
a
100% - V%
Where T is the adjusted total test duration in hours, t is the original test duration in hours, and V percent is the largest
a o
percentage of voltage reduction made in any specified voltage.
METHOD 1005.8
1 June 1993
4

3.5.6.1 Special considerations for devices with internal thermal limitation. For devices with internal thermal shutdown,
extended exposure at a temperature in excess of the shut-down temperature will not provide a realistic indicator of long-term
operating reliability. For such devices, measurement of the case temperature should be made at the specified bias voltages
at several different ambient temperatures. From these measurements, junction temperatures should be computed, and the
operating life shall be performed at that ambient temperature which, with the voltage biases specified, will result in a worst
case junction temperature at least 5°C but no more than 10°C below the minimum junction temperature at which the device
would go into thermal shutdown, and the test time shall be determined from table I for the applicable device class level.
4. SUMMARY. The following details shall be specified in the applicable acquisition document:
a. Special preconditioning, when applicable.
b. Test temperature, and whether ambient, junction, or case, if other than as specified in 3.2.
c. Test duration, if other than as specified in 3.1.
d. Test mounting, if other than normal (see 3).
e. Test condition letter.
f. End-point measurements and intermediate measurements (see 3.3).
g. Criteria for device failure for intermediate and end-point measurements (see 3.3), if other than device
specification limits, and criteria for lot acceptance.
h. Test sample (see 3.4).
i. Time to complete end-point measurements, if other than as specified (see 3.3).
j. Authorization for use of condition F and special maximum test rating for condition F, when applicable (see 4.b).
k. Time temperature conditions for condition F, if other than as specified in 3.5.6.
METHOD 1005.8
1 June 1993
5

TABLE I. Steady-state time temperature regression. 1/ 2/ 3/ 4/
Minimum Minimum time (hours) Test
temperature condition
TA (°C ) (see 3.5)
Class level Class level Class level S
S B hybrids
(Class K)
100 7500 7500 Hybrid only
105 4500 4500 "
110 3000 3000 "
115 2000 2000 "
120 1500 1500 "
125 1000 1000 1000 A -E
130 900 704 --- "
135 800 496 --- "
140 700 352 --- "
145 600 256 --- "
150 500 184 --- "
175 40 --- F
180 32 --- "
185 31 --- "
190 30 --- "
1/ Test condition F shall be authorized prior to use and consists of temperatures 175°C
and higher.
2/ For condition F the maximum junction temperature is unlimited and care shall be taken
to ensure the device(s) does not go into thermal runaway.
3/ The only allowed conditions are as stated above.
4/ Test temperatures below 125°C may be used for hybrid circuits only.
METHOD 1005.8
1 June 1993
6

| Minimum temperature TA (°C ) | Minimum time (hours) |  |  | Test condition (see 3.5) |
| --- | --- | --- | --- | --- |
|  | Class level S | Class level B | Class level S hybrids (Class K) |  |
| 100 |  | 7500 | 7500 | Hybrid only |
| 105 |  | 4500 | 4500 | " |
| 110 |  | 3000 | 3000 | " |
| 115 |  | 2000 | 2000 | " |
| 120 |  | 1500 | 1500 | " |
| 125 | 1000 | 1000 | 1000 | A -E |
| 130 | 900 | 704 | --- | " |
| 135 | 800 | 496 | --- | " |
| 140 | 700 | 352 | --- | " |
| 145 | 600 | 256 | --- | " |
| 150 | 500 | 184 | --- | " |
| 175 |  | 40 | --- | F |
| 180 |  | 32 | --- | " |
| 185 |  | 31 | --- | " |
| 190 |  | 30 | --- | " |

FIGURE 1005-1. Steady-state.
FIGURE 1005-2. Typical parallel, series excitation.
NOTE: For free running counter, N is an odd number and the output of N is connect to the input of 1.
FIGURE 1005-3. Ring oscillator.
METHOD 1005.8
1 June 1993
7

METHOD 1005.8
1 June 1993
8
