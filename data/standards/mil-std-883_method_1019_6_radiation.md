---
standard: MIL-STD-883
method: "1019.6"
category: radiation
language: en
---

# MIL-STD-883 Method 1019.6 — IONIZING RADIATION (TOTAL DOSE) TEST PROCEDURE

METHOD 1019.6
IONIZING RADIATION (TOTAL DOSE) TEST PROCEDURE
1. PURPOSE. This test procedure defines the requirements for testing packaged semiconductor integrated circuits for
ionizing radiation (total dose) effects from a cobalt-60 (60Co) gamma ray source. The testing includes both standard room
temperature irradiation as well as irradiation at elevated temperature. In addition this procedure provides an accelerated
annealing test for estimating low dose rate ionizing radiation effects on devices. This annealing test is important for low
dose-rate or certain other applications in which devices may exhibit significant time-dependent effects. This procedure
addresses only steady state irradiations, and is not applicable to pulse type irradiations. This test may produce severe
degradation of the electrical properties of irradiated devices and thus should be considered a destructive test.
1.1 Definitions. Definitions of terms used in this procedure are given below:
a. Ionizing radiation effects. The changes in the electrical parameters of a device or integrated circuit resulting from
radiation-induced charge. These are also referred to as total dose effects.
b. In-flux test. Electrical measurements made on devices during irradiation exposure.
c. Not in-flux test. Electrical measurements made on devices at any time other than during irradiation.
d. Remote tests. Electrical measurements made on devices which are physically removed from the radiation
location.
e. Time dependent effects. Significant degradation in electrical parameters caused by the growth or annealing or
both of radiation-induced trapped charge after irradiation. Similar effects also take place during irradiation.
f. Accelerated annealing test. A procedure utilizing elevated temperature to accelerate time-dependent effects.
g. Enhanced Low Dose Rate Sensitivity (ELDRS). Used to refer to a part that shows enhanced radiation induced
damage at dose rates below 50 rad(Si)/s.
h. Overtest. A factor that is applied to the specification dose to determine the test dose level that the samples must
pass to be acceptable at the specification level. An ovetest factor of 1.5 means that the parts must be tested at 1.5
times the specification dose.
i. Parameter Delta Design Margin (PDDM). A design margin that is applied to te radiation induced change in an
electrical parameter. For a PDDM of 2 the change in a parameter at a specified dose from the pre-irradiation value
is multiplied by two and added to the post-irradiation value to see if the sample exceeds the post-irradiation
parameter limit. For example, if the pre-irradiation value of Ib is 30 nA and the post-irradiation value at 20 krad(Si)
is 70 nA (change in Ib is 40 nA), then for a PDDM of 2 the post-irradiation value would be 110 nA (30 nA + 2 X 40
nA). If the allowable post-irradiation limit is 100 nA the part would fail.
2. APPARATUS. The apparatus shall consist of the radiation source, electrical test instrumentation, test circuit board(s),
cabling, interconnect board or switching system, an appropriate dosimetry measurement system, and an environmental
chamber (if required for time-dependent effects measurements or elevated temperature irradiation). Adequate precautions
shall be observed to obtain an electrical measurement system with sufficient insulation, ample shielding, satisfactory
grounding, and suitable low noise characteristics.
2.1 Radiation source. The radiation source used in the test shall be the uniform field of a 60Co gamma ray source.
Uniformity of the radiation field in the volume where devices are irradiated shall be within ±10 percent as measured by the
dosimetry system, unless otherwise specified. The intensity of the gamma ray field of the 60Co source shall be known with
an uncertainty of no more than ±5 percent. Field uniformity and intensity can be affected by changes in the location of the
device with respect to the radiation source and the presence of radiation absorption and scattering materials.
METHOD 1019.6
7 March 2003
1

2.2 Dosimetry system. An appropriate dosimetry system shall be provided which is capable of carrying out the
measurements called for in 3.2. The following American Society for Testing and Materials (ASTM) standards and guidelines
or other appropriate standards and guidelines shall be used:
ASTM E 666 - Standard Method for Calculation of Absorbed Dose from Gamma or X Radiation.
ASTM E 668 - Standard Practice for the Application of Thermoluminescence Dosimetry (TLD) Systems
for Determining Absorbed Dose in Radiation-Hardness Testing of Electronic Devices.
ASTM E 1249 - Minimizing Dosimetry Errors in Radiation Hardness Testing of Silicon Electronic Devices.
ASTM E 1250 - Standard Method for Application of Ionization Chambers to Assess the Low Energy Gamma
Component of Cobalt 60 Irradiators Used in Radiation Hardness Testing of Silicon Electronic
Devices.
ASTM E 1275 - Standard Practice for Use of a Radiochromic Film Dosimetry System.
ASTM F 1892 - Standard Guide for Ionizing Radiation (Total Dose) Effects Testing of Semiconductor Devices.
These industry standards address the conversion of absorbed dose from one material to another, and the proper use of
various dosimetry systems. 1/
2.3 Electrical test instruments. All instrumentation used for electrical measurements shall have the stability, accuracy,
and resolution required for accurate measurement of the electrical parameters. Any instrumentation required to operate in a
radiation environment shall be appropriately shielded.
2.4 Test circuit board(s). Devices to be irradiated shall either be mounted on or connected to circuit boards together with any
associated circuitry necessary for device biasing during irradiation or for in-situ measurements. Unless otherwise specified, all
device input terminals and any others which may affect the radiation response shall be electrically connected during irradiation,
i.e., not left floating. The geometry and materials of the completed board shall allow uniform irradiation of the devices under test.
Good design and construction practices shall be used to prevent oscillations, minimize leakage currents, prevent electrical
damage, and obtain accurate measurements. Only sockets which are radiation resistant and do not exhibit significant leakages
(relative to the devices under test) shall be used to mount devices and associated circuitry to the test board(s). All apparatus
used repeatedly in radiation fields shall be checked periodically for physical or electrical degradation. Components which are
placed on the test circuit board, other than devices under test, shall be insensitive to the accumulated radiation or they shall be
shielded from the radiation. Test fixtures shall be made such that materials will not perturb the uniformity of the radiation field
intensity at the devices under test. Leakage current shall be measured out of the radiation field. With no devices installed in the
sockets, the test circuit board shall be connected to the test system such that all expected sources of noise and interference are
operative. With the maximum specified bias for the test device applied, the leakage current between any two terminals shall not
exceed ten percent of the lowest current limit value in the pre-irradiation device specification. Test circuit boards used to bias
devices during accelerated annealing must be capable of withstanding the temperature requirements of the accelerated
annealing test and shall be checked before and after testing for physical and electrical degradation.
2.5 Cabling. Cables connecting the test circuit boards in the radiation field to the test instrumentation shall be as short as
possible. If long cables are necessary, line drivers may be required. The cables shall have low capacitance and low
leakage to ground, and low leakage between wires.
2.6 Interconnect or switching system. This system shall be located external to the radiation environment location, and
provides the interface between the test instrumentation and the devices under test. It is part of the entire test system and
subject to the limitation specified in 2.4 for leakage between terminals.
2.7 The environmental chamber. The environmental chamber for time-dependent effects testing, if required, shall be
capable of maintaining the selected accelerated annealing temperature within ±5°C.
1/ Copies may be obtained from the American Society for Testing and Materials, 1916 Race Street, Philadelphia, PA 19103.
METHOD 1019.6
7 March 2003
2

2.8 The irradiation temperature chamber. The irradiation temperature, if required for elevated temperature irradiation should
be capable of maintaining a circuit under test at 100 °C + 5 °C while it is being irradiated. The chamber should be capable of
raising the temperature of the circuit under test from room temperature to the irradiation temperature within a reasonable time
prior to irradiation and cooling the circuit under test from the irradiation temperature to room temperature in less than 20 minutes
following irradiation. The irradiation bias shall be maintained during the heating and cooling. The method for raising,
maintaining and lowering the temperature of the circuit under test may be by conduction through a heat sink using heating and
cooling fluids, by convection using forced hot and cool air, or other means that will achieve the proper results.
3. PROCEDURE. The test devices shall be irradiated and subjected to accelerated annealing testing (if required for time-
dependent effects testing) as specified by a test plan. This plan shall specify the device description, irradiation conditions,
device bias conditions, dosimetry system, operating conditions, measurement parameters and conditions, and accelerated
annealing test conditions (if required).
3.1 Sample selection and handling. Only devices which have passed the electrical specifications as defined in the test plan
shall be submitted to radiation testing. Unless otherwise specified, the test samples shall be randomly selected from the parent
population and identically packaged. Each part shall be individually identifiable to enable pre- and post-irradiation comparison.
For device types which are ESD-sensitive, proper handling techniques shall be used to prevent damage to the devices.
3.2 Burn-in. For some devices, there are differences in the total dose radiation response before and after burn-in.
Unless it has been shown by prior characterization or by design that burn-in has negligible effect (parameters remain within
postirradiation specified electrical limits) on the total dose radiation response, then one of the following must be done:
3.2.1 The manufacturer shall subject the radiation samples to the specified burn-in conditions prior to conducting total
dose radiation testing or
3.2.2 The manufacturer shall develop a correction factor, (which is acceptable to the parties to the test) taking into
account the changes in total dose response resulting from subjecting product to burn-in. The correction factor shall then be
used to accept product for total dose response without subjecting the test samples to burn-in.
3.3 Dosimetry measurements. The radiation field intensity at the location of the device under test shall be determined
prior to testing by dosimetry or by source decay correction calculations, as appropriate, to assure conformance to test level
and uniformity requirements. The dose to the device under test shall be determined one of two ways: (1) by measurement
during the irradiation with an appropriate dosimeter, or (2) by correcting a previous dosimetry value for the decay of the 60Co
source intensity in the intervening time. Appropriate correction shall be made to convert from the measured or calculated
dose in the dosimeter material to the dose in the device under test.
3.4 Lead/Aluminum (Pb/Al) container. Test specimens shall be enclosed in a Pb/Al container to minimize dose enhancement
effects caused by low-energy, scattered radiation. A minimum of 1.5 mm Pb, surrounding an inner shield of at least 0.7 mm Al, is
required. This Pb/Al container produces an approximate charged particle equilibrium for Si and for TLDs such as CaF2. The radiation
field intensity shall be measured inside the Pb/Al container (1) initially, (2) when the source is changed, or (3) when the orientation or
configuration of the source, container, or test-fixture is changed. This measurement shall be performed by placing a dosimeter (e.g., a
TLD) in the device-irradiation container at the approximate test-device position. If it can be demonstrated that low energy scattered
radiation is small enough that it will not cause dosimetry errors due to dose enhancement, the Pb/Al container may be omitted.
3.5 Radiation level(s). The test devices shall be irradiated to the dose level(s) specified in the test plan within ±10
percent. If multiple irradiations are required for a set of test devices, then the post-irradiation electrical parameter
measurements shall be performed after each irradiation.
3.6 Radiation dose rate. The radiation dose rate for bipolar and BiCMOS linear or mixed-signal parts used in applications
where the maximum dose rate is below 50 rad(Si)/s shall be determined as described in paragraph 3.13 below. Parts used
in low dose rate applications, unless they have been demonstrated to not exhibit an ELDERS response shall use Condition
C, Condition D, or Condition E.
NOTE: Devices that contain both MOS and bipolar devices may require qualification to multiple subconditions to ensure that both
ELDRS and traditional MOS effects are evaluated.
3.6.1 Condition A. For condition A (standard condition) the dose rate shall be between 50 and 300 rad(Si)/s [0.5 and 3
Gy(Si)/s] 60Co 2/ The dose rates may be different for each radiation dose level in a series; however, the dose rate shall not
vary by more than ±10 percent during each irradiation.
2/ The SI unit for the quantity absorbed dose is the gray, symbol GY. 100 rad = 1 Gy.
METHOD 1019.6
7 March 2003
3

3.6.2 Condition B. For condition B, for MOS devices only, if the maximum dose rate is < 50 rad(Si)/s in the intended
application, the parties to the test may agree to perform the test at a dose rate ≥ the maximum dose rate of the intended
application. Unless the exclusions in 3.12.1b are met, the accelerated annealing test of 3.12.2 shall be performed.
3.6.3 Condition C. For condition C, (as an alternative) the test may be performed at the dose rate agreed to by the
parties to the test.
3.6.4 Condition D. For condition D, for bipolar or BiCMOS linear or mixed-signal circuits only, the parts shall be
irradiated at <10 mrad(Si)/s unless the specification dose is greater than 25 krad(Si). For radiation levels greater than 25
krad(Si) the total irradiation time shall be > 1000 hours and the dose rate shall be determined from the total dose (including
any overtest factors) and the irradiation time.
3.6.5 Condition E. For condition E, for bipolar or BiCMOS linear or mixed-signal circuits only, the parts shall be
irradiated at between 0.5 and 5 rad(Si)/s if the specification dose is < 50 krad(Si). Condition E applies to elevated
temperature irradiation at 100°C ±5°C and does not apply for devices with specification doses >50 krad(Si) unless it can be
demonstrated that the elevated temperature irradiation test provides a conservative bound for low dose rate response at a
radiation specification level that is above 50 krad(Si).
3.7 Temperature requirements. The following requirements shall apply for room temperature and elevated temperature
irradiation.
3.7.1 Room temperature irradiation. Since radiation effects are temperature dependent, devices under test shall be
irradiated in an ambient temperature of 24°C ±6°C as measured at a point in the test chamber in close proximity to the test
fixture. The electrical measurements shall be performed in an ambient temperature of 24°C ±6°C. If devices are
transported to and from a remote electrical measurement site, the temperature of the test devices shall not be allowed to
increase by more than 10°C from the irradiation environment. If any other temperature range is required, it shall be
specified.
3.7.2 Elevated temperature irradiation. For bipolar or BiCMOS linear or mixed-signal circuits irradiated using Condition E
dose rate, devices under test shall be irradiated in an ambient temperature of 100°C ±5°C as measured at a point in the test
chamber in close proximity to the test fixture.
3.8 Electrical performance measurements. The electrical parameters to be measured and functional tests to be
performed shall be specified in the test plan. As a check on the validity of the measurement system and pre- and post-
irradiation data, at least one control sample shall be measured using the operating conditions provided in the governing
device specifications. For automatic test equipment, there is no restriction on the test sequence provided that the rise in the
device junction temperature is minimized. For manual measurements, the sequence of parameter measurements shall be
chosen to allow the shortest possible measurement period. When a series of measurements is made, the tests shall be
arranged so that the lowest power dissipation in the device occurs in the earliest measurements and the power dissipation
increases with subsequent measurements in the sequence.
The pre- and post-irradiation electrical measurements shall be done on the same measurement system and the same
sequence of measurements shall be maintained for each series of electrical measurements of devices in a test sample.
Pulse-type measurements of electrical parameters should be used as appropriate to minimize heating and subsequent
annealing effects. Devices which will be subjected to the accelerated annealing testing (see 3.12) may be given a
preirradiation burn-in to eliminate burn-in related failures.
3.9 Test conditions. The use of in-flux or not in-flux testing shall be specified in the test plan. (This may depend on the
intended application for which the data are being obtained.) The use of in-flux testing may help to avoid variations
introduced by post-irradiation time dependent effects. However, errors may be incurred for the situation where a device is
irradiated in-flux with static bias, but where the electrical testing conditions require the use of dynamic bias for a significant
fraction of the total irradiation period. Not-in-flux testing generally allows for more comprehensive electrical testing, but can
be misleading if significant post-irradiation time dependent effects occur.
3.9.1 In-flux testing. Each test device shall be checked for operation within specifications prior to being irradiated. After
the entire system is in place for the in-flux radiation test, it shall be checked for proper interconnections, leakage (see 2.4),
and noise level. To assure the proper operation and stability of the test setup, a control device with known parameter values
shall be measured at all operational conditions called for in the test plan. This measurement shall be done either before the
insertion of test devices or upon completion of the irradiation after removal of the test devices or both.
3.9.2 Remote testing. Unless otherwise specified, the bias shall be removed and the device leads placed in conductive
foam (or similarly shorted) during transfer from the irradiation source to a remote tester and back again for further irradiation.
This minimizes post-irradiation time dependent effects.
METHOD 1019.6
7 March 2003
4

3.9.3 Bias and loading conditions. Bias conditions for test devices during irradiation or accelerated annealing shall be
within ±10 percent of those specified by the test plan. The bias applied to the test devices shall be selected to produce the
greatest radiation induced damage or the worst-case damage for the intended application, if known. While maximum
voltage is often worst case some bipolar linear device parameters (e.g. input bias current or maximum output load current)
exhibit more degradation with 0 V bias. The specified bias shall be maintained on each device in accordance with the test
plan. Bias shall be checked immediately before and after irradiation. Care shall be taken in selecting the loading such that
the rise in the junction temperature is minimized.
3.10 Post-irradiation procedure. Unless otherwise specified, the following time intervals shall be observed:
a. The time from the end of an irradiation to the start of electrical measurements shall be a maximum of 1 hour unless
Condition D is used, in which case the maximum time shall be 72 hours.
b. The time to perform the electrical measurements and to return the device for a subsequent irradiation, if any, shall
be within two hours of the end of the prior irradiation unless Condition D is used, in which case the maximum time
shall be 120 hours.
To minimize time dependent effects, these intervals shall be as short as possible. The sequence of parameter
measurements shall be maintained constant throughout the tests series.
3.11 Extended room temperature anneal test. The tests of 3.1 through 3.10 are known to be overly conservative for
some devices in a very low dose rate environment (e.g. dose rates characteristic of space missions). The extended room
temperature anneal test provides an estimate of the performance of a device in a very low dose rate environment even
though the testing is performed at a relatively high dose rate (e.g. 50-300 rad(Si)/s). The procedure involves irradiating the
device per steps 3.1 through 3.10 and post-irradiation subjecting the device under test to a room temperature anneal for an
appropriate period of time (see 3.11.2c) to allow leakage-related parameters that may have exceeded their pre-irradiation
specification to return to within specification. The procedure is known to lead to a higher rate of device acceptance in cases:
a. where device failure when subjected to the tests in 3.1 through 3.10 has been caused by the buildup of trapped
positive charge in relatively soft oxides, and
b. where this trapped positive charge anneals at a relatively high rate.
3.11.1 Need to perform an extended room temperature anneal test. The following criteria shall be used to determine
whether an extended room temperature anneal test is appropriate:
a. The procedure is appropriate for either MOS or bipolar technology devices.
b. The procedure is appropriate where only parametric failures (as opposed to functional failure) occurs. The parties
to the test shall take appropriate steps to determine that the device under test is subject to only parametric failure
over the total ionizing dose testing range.
c. The procedure is appropriate where the natural annealing response of the device under test will serve to correct the
out-of-specification of any parametric response. Further, the procedure is known to lead to a higher rate of device
acceptance in cases where the expected application irradiation dose rate is sufficiently low that ambient
temperature annealing of the radiation induced trapped positive charge can lead to a significant improvement of
device behavior. Cases where the expected application dose rate is lower than the test dose rate and lower than
0.1 rad(Si)/s should be considered candidates for the application of this procedure. The parties to the test shall
take appropriate steps to determine that the technology under test can provide the required annealing response
over the total ionizing dose testing range.
3.11.2 Extended room temperature anneal test procedure. If the device fails the irradiation and testing specified in 3.1
through 3.10, an additional room temperature annealing test may be performed as follows:
a. Following the irradiation and testing of 3.1 through 3.10, subject the device under test to a room temperature
anneal under worst-case static bias conditions. For information on worst case bias see 3.9.3,
METHOD 1019.6
7 March 2003
5

b. The test will be carried out in such a fashion that the case of the device under test will have a temperature within
the range 24°C ± 6°C.
c. Where possible, the room temperature anneal should continue for a length of time great enough to allow device
parameters that have exceeded their pre-irradiation specification to return to within specification or post-irradiation-
parametric limit (PIPL) as established by the manufacturer. However, the time of the room temperature anneal
shall not exceed t , where
max
D
spec
t =
max
R
max
D is the total ionizing dose specification for the part and R is the maximum dose rate for the intended use.
spec max
d. Test the device under test for electrical performance as specified in 3.7 and 3.8. If the device under test passes
electrical performance tests following the extended room temperature anneal, this shall be considered acceptable
performance for a very low dose rate environment in spite of having previously failed the post-irradiation and
electrical tests of 3.1 through 3.10.
3.12 MOS accelerated annealing test. The accelerated annealing test provides an estimate of worst-case degradation of
MOS microcircuits in low dose rate environments. The procedure involves heating the device following irradiation at
specified temperature, time and bias conditions. An accelerated annealing test (see 3.12.2) shall be performed for cases
where time dependent effects (TDE) can cause a device to degrade significantly or fail. Only standard testing shall be
performed as specified in 3.1 through 3.10 for cases where TDE are known not to cause significant device degradation or
failure (see 3.12.1) or where they do not need to be considered, as specified in 3.12.1.
3.12.1 Need to perform accelerated annealing test. The parties to the test shall take appropriate steps to determine
whether accelerated annealing testing is required. The following criteria shall be used:
a. The tests called out in 3.12.2 shall be performed for any device or circuit type that contains MOS circuit elements
(i.e., transistors or capacitors).
b. TDE tests may be omitted if:
1. circuits are known not to contain MOS elements by design, or
2. the ionizing dose in the application, if known, is below 5 krad(Si), or
3. the lifetime of the device from the onset of the irradiation in the intended application, if known, is short
compared with TDE times, or
4. the test is carried out at the dose rate of the intended application, or
5. the device type or IC technology has been demonstrated via characterization testing not to exhibit TDE
changes in device parameters greater than experimental error (or greater than an otherwise specified upper
limit) and the variables that affect TDE response are demonstrated to be under control for the specific vendor
processes.
At a minimum, the characterization testing in (5) shall include an assessment of TDE on propagation delay,
output drive, and minimum operating voltage parameters. Continuing process control of variables affecting
TDE may be demonstrated through lot sample tests of the radiation hardness of MOS test structures.
c. This document provides no guidance on the need to perform accelerated annealing tests on technologies that do
not include MOS circuit elements.
METHOD 1019.6
7 March 2003
6

3.12.2 Accelerated annealing test procedure. If the device passes the tests in 3.1 through 3.10 or if it passes 3.11 (if that
procedure is used) to the total ionizing dose level specified in the test plan or device specification or drawing and the
exclusions of 3.12.1 do not apply, the accelerated annealing test shall be conducted as follows:
a. Overtest.
1. Irradiate each test device to an additional 0.5-times the specified dose using the standard test conditions (3.1
through 3.10). Note that no electrical testing is required at this time.
2. The additional 0.5-times irradiation in 3.12.2.a.1may be omitted if it has been demonstrated via
characterization testing that:
a. none of the circuit propagation delay, output drive, and minimum operating voltage parameters recover
toward their pre-irradiation value greater than experimental accelerated annealing test of 3.12.2.b, and
b. the irradiation biases chosen for irradiation and accelerated annealing tests are worst-case for the
response of these parameters during accelerated annealing.
The characterization testing to establish worst-case irradiation and annealing biases shall be performed at the
specified level. The testing shall at a minimum include separate exposures under static and dynamic
irradiation bias, each followed by worst-case static bias during accelerated annealing according to 3.12.2.b.
b. Accelerated annealing. Heat each device under worst-case static bias conditions in an environmental chamber
according to one of the following conditions:
1. At 100°C ±5°C for 168 ±12 hours, or
2. At an alternate temperature and time that has been demonstrated via characterization testing to cause equal
or greater change in the parameter(s) of interest, e.g., propagation delay, output drive, and minimum
operating voltage, in each test device as that caused by 3.12.2.b.1, or
3. At an alternate temperature and time which will cause trapped hole annealing of >60% and interface state
annealing of <10% as determined via characterization testing of NMOS test transistors from the same
process. It shall be demonstrated that the radiation response of test transistors represent that of the device
under test.
c. Electrical testing. Following the accelerated annealing, the electrical test measurements shall be performed as
specified in 3.8 and 3.9.
METHOD 1019.6
7 March 2003
7

3.13 Test procedure for Bipolar and BiCMOS linear or mixed signal circuits with agreed to dose rates of less than 50
rad(Si)/s. Many bipolar linear parts exhibit ELDRS, which cannot be simulated with a room temperature 50-300 rad(Si)/s
irradiation plus elevated temperature anneal, such as that used for MOS parts (see ASTM-F1892 for more technical details).
Parts that exhibit ELDRS shall be tested either at the agreed to dose rate, at a prescribed low dose rate to an overtest
radiation level, or with an elevated temperature irradiation test that includes a parameter delta design margin.
Need to perform low dose rate testing.
a. The low dose rate tests described in 3.13 may be omitted if:
1. circuits are known not to contain bipolar transistors by design, or
2. circuits are known not to contain any linear circuit functions by design.
3. the device type and IC technology have been demonstrated via characterization testing not to exhibit
ELDRS in device parameters greater than experimental error (or greater than an otherwise specified
upper limit) and the variables that affect ELDRS response are demonstrated to be under control for the
specific vendor processes.
3.13.1 Low dose rate or elevated temperature irradiation test for bipolar or BiCMOS linear or mixed-signal circuits. All
circuits that do not meet the exception of 3.13.a shall be tested using one of the following test conditions.
Note: The test procedures in paragraphs b. and c. below represent a compromise between the desire for a conservative,
worst-case test and the constraints of test cost, schedule and facilities. For this reason, the test procedures may
result in a non-conservative test for some kinds of circuits.
a. Test at the agreed to dose rate. Irradiate each test device at the dose rate described in 3.6.3 Condition C using
the standard test conditions (3.1 through 3.10).
b. Test at a prescribed low dose rate. Irradiate each test device at the close rate described in 3.6.4 Condition D
using the standard test conditions (3.1 through 3.10) with the following additional requirements:
1. If the dose rate is < 10 mrad(Si)/s, an overtest factor of 1.5 shall be applied to the radiation test level, i.e.
the part must pass at a radiation level of 1.5 times the specification dose to be acceptable.
2. If the dose rate is greater than 10 mrad(Si)/s, an overtest factor of 2.0 shall be applied to the radiation
test level, i.e. the part must pass at a radiation level of 2.0 times the specification dose to be acceptable.
c. Test at an elevated temperature. Irradiate each test device at the dose rate described in 3.6.5 Condition E using
the standard test conditions (3.1 through 3.10) with the following additional requirements:
1. The irradiation temperature shall be 100°C ± 5°C using an irradiation test chamber as described in
paragraph 2.8. Every effort shall be made to minimize the time at temperature.
2. The elevated temperature irradiation test shall only be used for parts with a specification dose of 50
krad(Si) or less.
3. All pre and post irradiation electrical measurements shall be made at a temperature of 24°C ±6°C.
4. A parameter design margin of 3 shall be applied at the specification dose to all critical electrical
parameters in the following manner. The change in each electrical parameter shall be calculated for
each sample at the specification dose. This change in the parameter shall be multiplied by 3 and added
(or subtracted) to the post irradiation parameter value for each sample. This value shall be compared to
the allowable degraded value of the parameter to determine whether the sample passes or fails the test.
METHOD 1019.6
7 March 2003
8

3.14 Test report. As a minimum, the report shall include the device type number, serial number, the manufacturer,
package type, controlling specification, date code, and any other identifying numbers given by the manufacturer. The bias
circuit, parameter measurement circuits, the layout of the test apparatus with details of distances and materials used, and
electrical noise and current leakage of the electrical measurement system for in-flux testing shall be reported using drawings
or diagrams as appropriate. Each data sheet shall include the test date, the radiation source used, the bias conditions
during irradiation, the ambient temperature around the devices during irradiation and electrical testing, the duration of each
irradiation, the time between irradiation and the start of the electrical measurements, the duration of the electrical
measurements and the time to the next irradiation when step irradiations are used, the irradiation dose rate, electrical test
conditions, dosimetry system and procedures and the radiation test levels. The pre- and post-irradiation data shall be
recorded for each part and retained with the parent population data in accordance with the requirements of MIL-PRF-38535
or MIL-PRF-38534. Any anomalous incidents during the test shall be fully documented and reported. The accelerated
annealing procedure, if used, shall be described. Any other radiation test procedures or test data required for the delivery
shall be specified in the device specification, drawing or purchase order.
4. SUMMARY. The following details shall be specified in the applicable acquisition document as required:
a. Device-type number(s), quantity, and governing specifications (see 3.1).
b. Radiation dosimetry requirements (see 3.3).
c. Radiation test levels including dose and dose rate (see 3.5 and 3.6).
d. Irradiation, electrical test and transport temperatures if other than as specified in 3.7.
e. Electrical parameters to be measured and device operating conditions during measurement (see 3.8).
f. Test conditions, i.e., in-flux or not-in-flux type tests (see 3.9).
g. Bias conditions for devices during irradiation (see 3.9.3).
h. Time intervals of the post-irradiation measurements (see 3.10).
i. Requirement for extended room temperature anneal test, if required (see 3.11).
j. Requirement for accelerated annealing test, if required (see 3.12).
k. Documentation required to be delivered with devices (see 3.14).
METHOD 1019.6
7 March 2003
9

SELECT DOSE RATE
SEE PARA. 3.6
IRRADIATE TO SPECIFIED DOSE
SEE PARA. 3.9
PERFORM SPECIFIED ELECTRICAL TESTS
SEE PARA. 3.8
FAIL
PASS
DETERMINE IF EXTENDED ROOM TEMPER-
ATURE ANNEAL TEST IS REQUIRED
SEE PARA. 3.11.1
PERFORM SPECIFIED
FAIL
ELECTRICAL TESTS
SEE PARA. 3.8
PASS
NO
DETERMINE IF ACCELERATED
PASS ANNEALING TEST IS REQUIRED
SEE PARA. 3.12.1
YES
DETERMINE IF 0.5X OVERTEST
IS REQUIRED
SEE PARA. 3.12.2.a.2
NO YES
IRRADIATE AN ADDITIONAL 0.5 X
SPECIFIED DOSE
SEE PARA. 3.12.2.a
PERFORM ONE OF THREE ACCELERATED
ANNEALING PROCEDURES
SEE PARA. 3.12.2.b
PASS FAIL
PERFORM SPECIFIED ELECTRICAL TESTS
SEE PARA. 3.8
FIGURE 1019-1. Flow diagram for ionizing radiation test procedure for MOS and digital bipolar circuits.
METHOD 1019.6
7 March 2003
10

|  | NO |
| --- | --- |
| PASS |  |

Determine the need for ELDRS testing
See Para. 3.13
No
Yes
Perform standard test
(Para 3.6.1 Condition A)
Pass Fail
See Para 3.13.1
Test at the intended Perform low dose rate test Perform elevated temperature
application dose rate per Para 3.6.4, Condition D irradiation per Para 3.6.5,
Para 3.6.3 Condition E
1. ≤ 25 krad:
Condition C
≤ 10 mrad/s 1. May be used if spec dose
dose = 1.5 spec ≤ 50 krad
Pass Fail 2. >25 krad: 2. 0.5 to 5 rad/s, 100 ° C
≥
1000 hrs 3. Parameter design margin = 3
dose = 2 spec
Pass Fail Pass Fail
Figure 1019-2. Flow diagram for ionizing radiation test procedure for bipolar (or BiCMOS) linear
or mixed-signal circuits
METHOD 1019.6
7 March 2003
11

METHOD 1019.6
7 March 2003
12
