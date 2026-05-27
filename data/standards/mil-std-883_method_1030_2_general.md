---
standard: MIL-STD-883
method: "1030.2"
category: general
language: en
---

# MIL-STD-883 Method 1030.2

METHOD 1030.2
*
PRESEAL BURN-IN
1. PURPOSE. The purpose of preseal burn-in is to identify marginal devices or stabilize monolithic, hybrid, or multichip
microcircuits prior to the sealing of packages so that rework or retrimmings can be performed. Standard or sealed-lid
burn-in testing (see method 1015) is designed to screen or eliminate marginal devices by stressing microcircuits at or above
maximum rated operating conditions or by applying equivalent screening conditions which will reveal time and stress failure
modes with equal or greater sensitivity. Performance of a portion of the standard burn-in testing prior to sealing will identify
marginal devices or those requiring retrimming at a point where rework or retrimming can readily be performed. Use of
preseal burn-in is optional and should be a function of the complexity of the microcircuit in question coupled with, if available,
actual sealed-lid burn-in failure rates.
2. APPARATUS. Details for the required apparatus and compensation for air velocity, when required, shall be as
described in method 1005. In addition, the oven used for preseal burn-in shall be so equipped to provide a dry (less than
100 ppm moisture, at the supply point) nitrogen at class 100,000 maximum environment. Suitable equipment shall be
provided to control the flow of dry nitrogen and to monitor the moisture content of the dry nitrogen flowing into the oven.
3. PROCEDURE. All microcircuits shall be subjected to the specified preseal burn-in test condition (see 3.1) for the time
and temperature and in the environment specified after all assembly operations, with the exception of lid sealing, have been
completed (see method 5004 herein, MIL-PRF-38534 or MIL-PRF-38535); internal visual inspection shall be performed prior
to sealing. The microcircuits shall be mounted by the leads, stud, or case in their normal mounting configuration, and the
point of connection shall be maintained at a temperature not less than the specified ambient temperature. Measurements
before and after preseal burn-in shall be made as specified.
3.1 Test conditions. Basic test conditions are as shown below. Details of each of these conditions shall be as described
in method 1005.
a. Test condition C: Steady-state dc voltages.
b. Test condition D: Series or parallel excitation with ac conditions as applicable to exercise the device under test to
normal operating conditions.
3.1.1 Test time. Unless otherwise specified, preseal burn-in shall be performed for a minimum of 48 hours. It shall be
permissible to divide the total minimum burn-in time between preseal and postseal burn-in provided that the total burn-in
time equals or exceeds the specified burn-in time of 160 hours and that the postseal burn-in time equals or exceeds 96
hours.
3.1.2 Test temperature. Unless otherwise specified, the preseal burn-in test temperature shall be 125°C. If a lower
temperature is used, a corresponding increase in time is necessary as shown on figure 1015-1.
* 3.1.3 Test environment. Preseal burn-in shall be performed in a dry nitrogen (less than 100 ppm moisture, at the supply
point), 100,000 (5 (cid:72)m or greater) particles/cubic foot controlled environment (class 8 of ISO 14644-1). Prior to heat-up, the
oven shall be purged with dry nitrogen and then the bias shall be applied. Testing shall not commence until the specified
environment has been achieved.
3.2 Measurements. Measurements before preseal burn-in, shall be conducted prior to applying preseal burn-in test
conditions. Unless otherwise specified, measurements after preseal burn-in shall be completed within 96 hours after
removal of the microcircuits from the specified pre-seal burn-in test condition and shall consist of all 25°C dc parameter
measurements (subgroup A-1 of method 5005) and all parameters for which delta limits have been specified as part of
interim electrical measurements. Delta limit acceptance, when applicable, shall be based upon these measurements. If
these measurements cannot be completed within 96 hours, the microcircuits shall be subjected to the same specified test
conditions (see 3.1) previously used for a minimum of 24 additional hours before measurements after pre-seal burn-in are
made.
METHOD 1030.2
18 June 2004
1

3.2.1 Cooldown after preseal burn-in. All microcircuits shall be cooled to within 10°C of their power stable condition at
room temperature prior to the removal of bias. The interruption of bias for up to 1 minute for the purpose of moving the
microcircuits to cool-down positions separate from the chamber within which testing was performed shall not be considered
removal of bias. Alternatively, except for linear or MOS devices (CMOS, NMOS, PMOS, etc.) the bias may be removed
during cooling provided the case temperature of microcircuits under test is reduced to a maximum of 35°C within 30 minutes
after the removal of the test conditions. All 25°C dc measurements shall be completed prior to any reheating of the
microcircuits.
3.2.2 Failure verification and repair. Microcircuits which fail the 25°C dc measurements after preseal burn-in shall be
submitted for failure verification in accordance with test condition A of method 5003. After verification and location of the
defective or marginal device in the microcircuit, rework shall be performed as allowed in MIL-PRF-38535 or
MIL-PRF-38534. Upon completion of rework, repaired microcircuits shall be remeasured and, if found satisfactory, shall be
returned for additional preseal burn-in (see 3.1) if such rework involved device replacement.
3.2.3 Test setup monitoring. The test setup shall be monitored at the test temperature initially and at the conclusion of
the test to establish that all microcircuits are being stressed to the specified requirements. The following is the minimum
acceptable monitoring procedure:
a. Device sockets. Initially and at least each 6 months thereafter, each test board or tray shall be checked to verify
continuity to connector points to assure that bias supplies and signal information will be applied to each socket.
Except for this initial and periodic verification, each microcircuit socket does not have to be checked; however,
random sampling techniques shall be applied prior to each time a board is used and shall be adequate to assure
that there are correct and continuous electrical connections to the microcircuits under test.
b. Connectors to test boards or trays. After the test boards are loaded with microcircuits and are inserted into the
oven, and prior to the nitrogen purge, each required test voltage and signal condition shall be verified in at least
one location on each test board or tray so as to assure electrical continuity and the correct application of specified
electrical stresses for each connection or contact pair used in the applicable test configuration.
c. At the conclusion of the test period, after cool-down, the voltage and signal condition verification of b above shall
be repeated.
Where failures or open contacts occur which result in removal of the required test stresses for any period of the required test
duration (see 3.1), the test time shall be extended to assure actual exposure for the total minimum specified test duration.
3.3 Handling of unsealed microcircuits. It is recommended that unsealed microcircuits be covered at all times for
protection from handling induced defects. Snap-on metal covers, or rigid plastic covers with a conductive coating, may be
removed from the microcircuits after all microcircuits are in place in the burn-in racks. Covers, if removed shall be replaced
immediately after bias removal and completion of burn-in and cool-down prior to removal of microcircuits from the burn-in
racks. Regardless of the method of handling during the time period between the completion of internal visual inspection
following preseal burn-in and sealing, the microcircuits shall be retained in a controlled environment (see method 2017).
3.4 Sealed-lid burn-in. After completion of preseal burn-in, internal visual, other preseal screens and sealing all
microcircuits shall undergo the screening specified in method 5004 herein, MIL-PRF-38534 or MIL-PRF-38535 except that
stabilization bake may be deleted. Sealed-lid burn-in shall be performed as specified in method 5004 herein , MIL-PRF-
38534 or MIL-PRF-38535 (see method 1015 for test details).
METHOD 1030.2
18 June 2004
2

4. SUMMARY. The following details shall be specified in the applicable acquisition document.
a. Test condition letter and burn-in circuit with requirements for inputs, outputs, applied voltages and power
dissipation as applicable (see 3.1).
b. Test mounting, if other than normal (see 3).
c. Pre and post preseal burn-in measurement and shift limits, as applicable (see 3.2).
d. Time within which post preseal burn-in measurements must be completed if other than specified (see 3.2).
e. Type of covers used to protect microcircuits from handling induced defects (see 3.3).
f. Test duration for preseal and sealed lid burn-in (see 3.1.1).
g. Test temperature, if less than 125°C (see 3.1.2).
METHOD 1030.2
18 June 2004
3

METHOD 1030.2
18 June 2004
4
