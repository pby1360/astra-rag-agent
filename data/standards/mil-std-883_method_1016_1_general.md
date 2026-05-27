---
standard: MIL-STD-883
method: "1016.1"
category: general
language: en
---

# MIL-STD-883 Method 1016.1 — LIFE/RELIABILITY CHARACTERIZATION TESTS

*
METHOD 1016.1
LIFE/RELIABILITY CHARACTERIZATION TESTS
1. PURPOSE. The purpose of the life characterization tests is to determine: (1) the life distributions, (2) the life
acceleration characteristics, and (3) the failure rate ((cid:27)) potential of the devices. For a discussion of failure rates and life
test considerations, see MIL-HDBK-217. Failure rates are ordinarily determined either for general qualification of devices or
the production lines from which they are obtained or for the purpose of predicting the failure rates (or Mean Time Between
Failure (MTBF)) of equipments in which the devices are to be employed.
* NOTE: A detailed dissertation on the life test result analysis techniques, with application examples, is presented by D.S.
Peck in the Proceedings of the 9th Annual Reliability Physics Symposium (1971), pages 69 through 78. Further
improvements to the methods of test result analysis are possible by using computer aided techniques such as
regression analysis and iterative curve fitting.
2. APPARATUS. Suitable sockets or other mounting means shall be provided to make firm electrical contact to the
terminals of devices under test in the specified circuit configuration. The mounting means shall be so designed that they will
not remove internally dissipated heat from the device by conduction, other than that removed through the device terminals
and the necessary electrical contacts, which shall be maintained at or above the specified ambient temperature. The
apparatus shall provide for maintaining the specified biases at the terminal(s) of the device under test and, when specified,
monitoring of the input excitation. Power supplies and current-setting resistors shall be capable of maintaining the specified
operating conditions, as minima, throughout the testing period with normal variations in their source voltages, ambient
temperatures, etc. The test equipment shall preferably be so arranged that only natural-convection cooling of the devices
occurs. When test conditions result in significant power dissipation, the test apparatus shall be arranged so as to result in
the approximate average power dissipation for each device whether devices are tested individually or in a group. The test
circuits need not compensate for normal variations in individual device characteristics but shall be arranged so that the
existence of failed or abnormal (i.e., open, short, etc.) devices in a group does not negate the effect of the test for other
devices in the group.
3. PROCEDURE. The microelectronic devices shall be subjected to the specified test condition (see 3.4) for the
specified duration and test temperature, and the required measurements shall be made at the specified intermediate points
and endpoints. Lead-, stud-, or case-mounted devices shall be mounted by the leads, stud, or case in their normal mounting
configuration, and the point of connection shall be maintained at a temperature not less than the specified temperature. The
test condition, duration, sample size, and temperature selected prior to test shall be recorded and shall govern for the entire
test.
3.1 Test duration. The life test duration shall be as follows:
Initial qualification: 4,000 (+72, -240) hours
or 75 percent failures, whichever comes first
All other tests: 1,000 (+72, -24) hours
or 50 percent failures, whichever comes first
Within the time interval of 24 hours before to 72 hours after the specified duration of the test, the devices shall be removed
from the specified test conditions and allowed to reach standard test conditions prior to the removal of bias.
3.2 Measurements. Measurements shall be grouped into two categories as follows:
Type A: Initial and final measurement.
Type B: Interim measurements.
METHOD 1016.1
18 June 2004
1

Unless otherwise specified, all measurements shall be completed within 8 hours after removal of the device from the
specified test conditions and shall consist of the following:
Type A: All specified endpoint measurement.
Type B: Selected critical parameters (see 4).
The type A measurements shall be made at the zero hour and final measurement time. The type B interim measurements
shall be made at the 4, 8, 16, 32, 64, 128, 256, 512 hour times for the 1000 hour test and additionally, at 1000, and 2000
hour times for the 4000 hour test.
3.2.1 Measurements following life test. When devices are measured following application of life test conditions, they shall
be cooled to room temperature prior to the removal of bias. All specified 25°C electrical measurements shall be completed
prior to any reheating of the devices.
3.2.2 Test setup monitoring. The test setup shall be monitored at the test temperature initially and at the conclusion of
the test to establish that all devices are being stressed to the specified requirements. The following is the minimum
acceptable monitoring procedure:
a. Device sockets. Initially and at least each 6 months thereafter, each test board or tray shall be checked to verify
continuity to connector points to assure that bias supplies and signal information will be applied to each socket.
Except for this initial and periodic verification, each device or device socket does not have to be checked; however,
random sampling techniques shall be applied prior to each time a board is used and shall be adequate to assure that
there are correct and continuous electrical connections to the devices under test.
b. Connectors to test boards or trays. After the test boards are loaded with devices, inserted into the oven, and brought
up to at least 125°C or the specified test temperature, whichever is less, each required test voltage and signal
condition shall be verified in at least one location on each test board or tray so as to assure electrical continuity and
the correct application of specified electrical stresses for each connection or contact pair used in the applicable test
configuration. This may be performed by opening the oven for a maximum of 10 minutes.
c. At the conclusion of the test period, prior to removal of devices from temperature and test conditions, the voltage and
signal condition verification of b above shall be repeated.
d. For class level S devices when loading boards or trays the continuity between each device and a bias supply shall be
verified.
Where failures or open contacts occur which result in removal of the required test stresses for any period of the required test
duration, the test time shall be extended to assure actual exposure for the total minimum specified test duration.
3.3 Test sample. The test sample shall be as specified (see 4). No fewer than 40 devices shall be specified for a given
test temperature.
3.4 Test conditions. In this condition microcircuits are subjected to bias(es) at temperatures (200°C to 300°C) which
considerably exceed the maximum rated operating temperature. At these elevated temperatures, it is generally found that
microcircuits will not operate normally as specified in their applicable acquisition document and it is therefore necessary that
special attention be given to the choice of bias circuits and conditions to assure that important circuit areas are adequately
biased, without damaging overstress of other areas of the circuit.
METHOD 1016.1
18 June 2004
2

3.4.1 Test temperatures. Unless otherwise specified, test temperatures shall be selected in the range of 200°C to 300°C.
The specified test temperature is the minimum actual ambient temperature to which all devices in the working area of the
chamber shall be exposed. This shall be assured by making whatever adjustments are necessary in the chamber profile,
loading, location of control or monitoring instruments, and the flow of air or other chamber atmosphere. Therefore,
calibration shall be accomplished on the chamber in a fully loaded, unpowered configuration, and the indicator sensor
located at, or adjusted to reflect the coldest point in the working area. For the initial failure rate determination test, three
temperatures shall be selected. A minimum of 25°C separation shall be maintained between the adjacent test temperatures
selected. All other periodic life tests shall be conducted with two temperatures and a minimum of 50°C separation.
3.4.2 Bias circuit selection. To properly select the accelerated test conditions, it is recommended that an adequate
sample of devices be exposed to the intended high temperature while measuring voltage(s) and current(s) at each device
terminal to assure that the applied electrical stresses do not induce damage. Therefore, prior to performing microcircuit life
tests, test circuit, thermal resistance (where significant), and step-stress evaluations should be performed over the test
ranges, usually 200°C to 300°C. Steps of 25°C for 24 hours minimum duration (all steps of equal duration with a tolerance
of no greater than ±5 percent), each followed by proper electrical measurements, shall be used for step-stress tests.
Optimum test conditions are those that provide maximum voltage at high thermal stress to the most failure-prone junctions
or sites, but maintain the device current at a controlled low level. Excessive device current may lead to thermal runaway
(and ultimately device destruction). Current-limiting resistors shall be employed.
The applied voltage at any or all terminal(s) shall be equal to the maximum rated voltage at 125°C. If necessary, only with
the specific approval of the qualifying activity, the applied voltage at any or all terminals(s) may be reduced to not less than
50 percent of the specified value(s) when it is demonstrated that excessive current flow or power dissipation would result
from operation at the specified voltage(s). If the voltage(s) is so reduced, the life testing time shall be determined in
accordance with the formula given in 3.5.6 of method 1005.
3.5 Life test ground rules. As an aid to selecting the proper test conditions for an effective microcircuit accelerated life or
screening test, the following rules have been formulated:
a. Apply maximum rated voltage (except as provided in 3.4.2) to the most failure prone microcircuit sites or junctions
identified during step-stress evaluation.
b. Apply electrical bias to the maximum number of junctions.
c. In each MOS or CMOS device, apply bias to different gate oxides so that both positive and negative voltages are
present.
d. Control device currents to avoid thermal runaway and excessive electromigration failures.
e. Employ current-limiting resistors in series with each device to ensure the application of electrical stress to all
nonfailed devices on test.
f. Select a value of each current-limiting resistor large enough to prevent massive device damage in the event of
failure, but small enough to minimize variations in applied voltage due to current fluctuations.
g. Avoid conditions that exceed design or material limitations such as solder melting points.
h. Avoid conditions that unduly accelerate nontypical field condition failure-mechanisms.
i. Employ overvoltage protection circuitry.
The determination of test conditions that conform to the established ground rules involves three basic steps: (1) evaluation
of candidate bias circuits at accelerated test temperatures, (2) device thermal characterization, and (3) the performance of
step-stress tests.
METHOD 1016.1
18 June 2004
3

3.6 Test results analysis. Failure analysis of the accelerated test results is necessary to separate the failures into
temperature and nontemperature dependent categories. The nontemperature dependent failures should be removed from
the test data prior to life distribution analysis. All failures shall be reported together with the analysis results and rationale for
deletion of those identified as nontemperature dependent.
3.6.1 Life distribution analysis. The effectiveness of the test result analysis can be enhanced by diligent failure analysis
of each test failure. Failures should be grouped by similarity of failure mechanisms, that is, surface related, metal migration,
intermetallic formation, etc. The time-to-failure history of each failure in a group should be recorded. This includes the
individual failure times and the associated calculated cumulative percent failures. To facilitate estimating the distribution
parameters from small-sample life tests, the data is plotted as a cumulative distribution. Since semiconductor life
distributions have been shown to follow a lognormal distribution, graph paper similar to figure 1016-1 is required for data
analysis. The lognormal distribution will appear as a straight line on this paper. The expected bimodal distribution of "freak"
and "main" populations in a combined form normally appears as an s-shaped plot. The distribution parameters necessary
for data analysis, median life and sigma (σ) can be calculated as:
timeof 50% failure
σ≈ ln
timeof 16% failure
Separate analysis of the individual "freak" and "main" populations should be performed and "goodness of fit" tests applied to
test the apparent distribution(s).
3.6.2 Life acceleration analysis. Life/reliability characterization requires the establishing of failure distributions for several
temperature stress levels at the same rated voltage condition. These failure distributions must represent a common failure
mechanism. Using a specially prepared graph paper for Arrhenius Reaction Rate Analysis as shown in figure 1016-2, the
median life times for the "freak" and "main" populations can be plotted to determine equivalent life-times at the desired use
temperatures.
3.6.3 Failure rate calculations. Semiconductor failures are lognormally distributed. Therefore, the failure rate will vary
with time. Semiconductor failure rates at any given time can be calculated using figure 1016-3 which is a normalized
presentation of the mathematical calculations for the instantaneous failure rate from a lognormal distribution.
4. SUMMARY. The following details shall be as specified in the applicable acquisition document:
a. Test temperature(s) and whether ambient or case.
b. Test mounting if other than normal (see 3).
c. Endpoint measurements (see 3.2).
d. Intermediate measurements (see 3.2).
e. Criteria for failure for endpoint and intermediate measurements (see 3.2), if other than device specification limits.
f. Test sample (see 3.3).
g. Requirements for inputs, outputs, biases, test circuit, and power dissipation, as applicable (see 3.4).
METHOD 1016.1
18 June 2004
4

h. Requirements for data analysis, including:
(1) Failure analysis results.
(2) Data calculations:
(a) Log normal by temperature.
(b) Reaction rate relationships
(c) Failure rate versus time.
METHOD 1016.1
18 June 2004
5

FIGURE 1016-1. Cumulative failure distribution plot.
METHOD 1016.1
18 June 2004
6

FIGURE 1016-2. Arrhenius plot - high temperature operating test - accelerated life.
METHOD 1016.1
18 June 2004
7

FIGURE 1016-3. Lognormal failure rates.
METHOD 1016.1
18 June 2004
8
)1T(
,EFIL
NAIDEM
=
EMIT
