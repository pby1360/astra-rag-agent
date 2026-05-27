---
standard: MIL-STD-883
method: "1032.1"
category: general
language: en
---

# MIL-STD-883 Method 1032.1 — PACKAGE INDUCED SOFT ERROR TEST PROCEDURE

METHOD 1032.1
PACKAGE INDUCED SOFT ERROR TEST PROCEDURE
(DUE TO ALPHA PARTICLES)
1. PURPOSE. This test method defines the procedure for testing integrated circuits under known test conditions for
susceptibility to alpha induced errors. This test was specifically designed to measure the device's ability to withstand alpha
particle impact. In addition, the procedure will determine the effectiveness of a "die-coating" shield. The test objective is to
determine the rate that failures are induced due to alpha radiation sourced from the device package, die and die-coat
material.
1.1 Definitions. The following definitions were created to be specific and relevant within the confines of this method.
1.1.1 DUT. Device under test.
1.1.2 Soft error. Any error induced by alpha particle impact resulting in either a transient error or an error in data storage
witnessed at the DUT's output.
1.1.3 Source. A foil of Thorium-232. (Note: This foil generates particles which have an alpha energy spectrum of 0
through 10 MeV).
1.1.4 Soft error rate (SER). Failures per unit time under normal conditions of package environment.
1.1.5 Accelerated soft error rate (ASER). Failures per unit time induced by exposure to a known alpha particle source.
1.1.6 Failure in time (FIT). 1 FIT = 1 failure in 109 device-hours.
1.1.7 Package flux. The total number of alpha particles impinging on the die surface per unit of time and area, due to
package material impurities (i.e., lid, die material, sealants, and optional alpha barrier material). Normal units of
measurement: alpha/cm2-hr.
1.1.8 Modified package flux. The total number of alpha particles impinging on the die surface per unit of time and area,
when a die coat is in place. Normal units of measurement: alpha/cm2-hr.
1.1.9 Source flux. The total number of alpha particles impinging on the die surface per unit of time and area, due to the
calibrated source. Normal units of measurement: alpha/cm2-s.
2. APPARATUS. The apparatus will consist of electrical test instrumentation, test circuit board(s), cabling, interconnect
boards, or switching systems and a Thorium-232 foil (optional). Precautions will be observed to obtain an electrical
measurement system with adequate shielding, low electrical noise induction, and proper grounding.
2.1 Radiation source. The radiation source used in this test shall be a Thorium-232 foil with dimensions large enough to
cover the entire exposed die cavity. The plated source shall be within the range of 0.01 - 5.0 µCi and shall produce the
same energy spectrum as the package impurities. Radiation sources must be controlled according to state and federal
regulations. The sources shall be certified periodically and decay rates used to determine the actual flux values at the time
of use. This source must be processed at least one year before being used. Caution: These sources should not be
exposed to heat.
2.2 Electrical test instruments. Electrical test instruments will be standard test instruments normally used for testing the
DUT. They must be capable of establishing the required test conditions and measuring the required electrical parameters.
All instruments shall be periodically calibrated in accordance with the general requirements of this test method standard.
METHOD 1032.1
29 May 1987
1

2.3 Test circuits. The test circuit shall contain the DUT, wiring, and auxiliary components as required. Connection will
allow for the application of the specified test conditions to obtain the specified outputs. Provision will be made for monitoring
and recording the specified outputs. Any loading of the output(s), such as resistors or capacitors, shall be specified. The
test circuit must not exhibit permanent changes in electrical characteristics as a result of exposure to the radioactive source.
Shielding will be incorporated to prevent such effects from occurring if necessary.
2.4 Cabling. Cabling, if required, shall be provided to connect the test circuit board containing the DUT to the test
instrumentation. All cables will be as short as possible. Care will be exercised to reduce electrical noise induced by the
cable by using shielded cable, triax, zipper tubing, or other shielding methods.
3. PROCEDURES. Two methods of testing are allowed by this procedure. The first is a long term test (sometimes
referred to as a system test) which does not incorporate a source but which accumulates a statistically valid amount of test
time to determine the SER directly. This method is self explanatory and must be accomplished using the same parameters
outlined in 3.1 (test plan). To determine the SER from this method, the following formula should be used and the result
converted to FIT's.
SER = Total number of errors/Total test time
The second method incorporates the use of the source outlined in 2.1 (radiation source). The procedure for testing with an
accelerated flux provided by the source is given below. These steps will be followed for each test outlined in 3.1.
a. The flux that the surface of the die would receive without a die coat will be determined. This is designated as the
package flux.
b. If the device has a die-coat it should be left in place for the next portion of the test. The DUT will be delidded and
the source placed directly over the die cavity at the same distance as the package lid was from the die.
NOTE: The distance between the foil and the die must be less than 50 mils and the foil must cover the entire die-
cavity opening in order to assure all angles of incidence will be maintained.
NOTE: If the DUT has an inverted die configuration (e.g., flip-chip) a test jig must be implemented which will
expose the active surface of the die to the irradiating source.
c. The testing outlined in 3.1 will be performed at this time with the configuration in b. above, in order to determine
the SER for each test performed.
d. Recorded for each test performed will be the following:
(1) Total number of errors recorded during each test.
(2) Time to accumulate the errors.
(3) SER , calculated from the following formulas:
1
ASER = Total number of errors/test time
1
SER = ASER x (Package flux/source flux)
1 1
e. If no die-coating has been applied, the SER will be reported as the measured rate of failure. However, if a die
1
coat exists, steps 3.f through 3.j will also be performed.
METHOD 1032.1
29 May 1987
2

f. The flux at the surface of the die will be determined when the die coat is in place; this is designated as the
modified package flux.
NOTE: The modified package flux should be the sum of the flux from the die and die-coat material only.
g. The die coat should be removed, assuring that no damage to the die has occurred and the source placed as
described in step b.
h. The tests performed in step 3.c must be repeated with this configuration, and the new SER will be designated
SER .
2
i. Recorded for each test performed will be the following:
(1) Total number of errors recorded during each test.
(2) Time to accumulate the errors.
(3) SER (SER ), calculated from the following formulas:
2
ASER = Total number of errors/test time
2
SER = ASER x (Modified package flux/source flux)
2 2
j. The SER for the corresponding tests will be summed and reported as the rate of failure for this DUT, using the
following formula:
SER = SER + SER
1 2
NOTE: The order of the steps above can be reversed to enable testing before the die coat is applied and then after it has
been applied, if desired.
3.1 Test plan. A test plan will be devised which will include determination of the worst case operating environment of the
DUT to determine the worst case SER, incorporating the steps outlined above. The data patterns used will ensure that each
cell and path, or both, is tested for both the logic zero and logic one states. The device will be continuously monitored and
refreshed and the data errors counted. This test will be required for each new device type or design revisions. The source
value and exposure time will be sufficient to obtain a significant number of soft error failures.
NOTE: If a data-retention or a reduced supply mode is a valid operating point for the DUT, this condition must also be
tested for its SER.
3.1.1 The test equipment program. The test equipment program will be devised to cycle and refresh the stored data or
cycled pattern continually, recording the number of errors.
3.1.2 Test conditions. Testing shall be performed at three separate cycle rates and at minimum and maximum voltages.
Unless otherwise specified, the following cycle timing will be used: The minimum and the maximum specified cycle timing
and the midpoint between the minimum and maximum specified cycle timing.
NOTE: If the device is a static or dynamic random access memory device, the device will be tested under both read and
write operations.
3.2 Report. As a minimum, the report will include device identification, test date, test operator, test facility (if applicable),
radiation source, test cycle times and voltages, data analysis, and equipment used in the test.
METHOD 1032.1
29 May 1987
3

4. SUMMARY. The following details shall be specified.
a. Device type and quantity to be tested.
b. Test circuit to be used.
c. Device output pins to be monitored.
d. Alpha source used if other than specified herein.
e. Alpha source Curie level.
f. Package flux measurement techniques.
g. Test equipment to be used.
h. Procedures for proper handling of radioactive materials.
METHOD 1032.1
29 May 1987
4
