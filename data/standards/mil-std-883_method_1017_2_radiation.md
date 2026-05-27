---
standard: MIL-STD-883
method: "1017.2"
category: radiation
language: en
---

# MIL-STD-883 Method 1017.2 — NEUTRON IRRADIATION

METHOD 1017.2
NEUTRON IRRADIATION
1. PURPOSE. The neutron irradiation test is performed to determine the susceptibility of semiconductor devices to
degradation in the neutron environment. The tests described herein are applicable to integrated circuits, transistors, and
diodes. This is a destructive test. Objectives of the test are: (1) to detect and measure the degradation of critical
semiconductor device parameters as a function of neutron fluence, and (2) to determine if specified semiconductor device
parameters are within specified limits after exposure to a specified level of neutron fluence (see section 4).
2. APPARATUS.
2.1 Test instruments. Test instrumentation to be used in the radiation test shall be standard laboratory electronic test
instruments such as power supplies, digital voltmeters, and picoammeters, etc., capable of measuring the electrical
parameters required. Parameter test methods and calibration shall be in accordance with MIL-STD-883 or MIL-STD-750,
whichever is applicable.
2.2 Radiation source. The radiation source used in the test shall be in a TRIGA Reactor or a Fast Burst Reactor.
Operation may be in either pulse or steady-state mode as appropriate. The source shall be one that is acceptable to the
acquiring activity.
2.3 Dosimetry equipment.
a. Fast-neutron threshold activation foils such as 32S, 54Fe, and 58Ni.
b. CaF thermoluminescence dosimeters (TLDs).
2
c. Appropriate activation foil counting and TLD readout equipment.
2.4 Dosimetry measurements.
2.4.1 Neutron fluences. The neutron fluence used for device irradiation shall be obtained by measuring
the amount of radioactivity induced in a fast-neutron threshold activation foil such as 32S, 54Fe, or 58Ni, irradiated
simultaneously with the device.
A standard method for converting the measured radioactivity in the specific activation foil employed into a neutron fluence is
given in the following Department of Defense adopted ASTM standards:
E263 - Standard Test Method for Measuring Fast-Neutron Reaction Rates by Radioactivation of
Iron.
E264 - Standard Test Method for Measuring Fast-Neutron Reaction Rates by Radioactivation of
Nickel.
E265 - Standard Test Method for Measuring Fast-Neutron Reaction Rates by Radioactivation of
Sulfur.
METHOD 1017.2
25 August 1983
1

The conversion of the foil radioactivity into a neutron fluence requires a knowledge of the neutron spectrum incident on the
foil. If the spectrum is not known, it shall be determined by use of the following DoD adopted ASTM standards, or their
equivalent:
E720 - Standard Guide for Selection of a Set of Neutron-Activation Foils for Determining Neutron Spectra
used in Radiation-Hardness Testing of Electronics.
E721 - Standard Method for Determining Neutron Energy Spectra with Neutron-Activation Foils for
Radiation-Hardness Testing of Electronics.
E722 - Standard Practice for Characterizing Neutron Energy Fluence Spectra in Terms of an Equivalent
Monoenergetic Neutron Fluence for Radiation-Hardness Testing of Electronics.
Once the neutron energy spectrum has been determined and the equivalent monoenergetic fluence calculated,
then an appropriate monitor foil (such as 32S, 54Fe, or 58Ni) should be used in subsequent irradiations to determine the
neutron fluence as discussed in E722. Thus, the neutron fluence is described in terms of the equivalent monoenergetic
neutron fluence per unit monitor response. Use of a monitor foil to predict the equivalent monoenergetic neutron fluence is
valid only if the energy spectrum remains constant.
2.4.2 Dose measurements. If absorbed, dose measurements of the gamma-ray component during the device test
irradiations are required, then such measurements shall be made with CaF thermoluminescence dosimeters (TLDs), or
2
their equivalent. These TLDs shall be used in accordance with the recommendations of the following DoD adopted ASTM
standard:
E668 - Standard Practice for the Application of Thermoluminescence-Dosimetry (TLD) Systems for
Determining Absorbed Dose in Radiation-Hardness Testing of Electronic Devices.
3. PROCEDURE.
3.1 Safety requirements. Neutron irradiated parts may be radioactive. Handling and storage of test specimens or
equipment subjected to radiation environments shall be governed by the procedures established by the local Radiation
Safety Officer or Health Physicist.
NOTE: The receipt, acquisition, possession, use, and transfer of this material after irradiation is subject to the regulations of
the U.S. Nuclear Regulatory Commission, Radioisotope License Branch, Washington, DC 20555. A by-product
license is required before an irradiation facility will expose any test devices. (U.S. Code, see 10 CFR 30-33.)
3.2 Test samples. A test sample shall be randomly selected and consist of a minimum of 10 parts, unless otherwise
specified. All sample parts shall have met all the requirements of the governing specification for that part. Each part shall
be serialized to enable pre and post test identification and comparison.
3.3 Pre-exposure.
3.3.1 Electrical tests. Pre-exposure electrical tests shall be performed on each part as required. Where delta parameter
limits are specified, the pre-exposure data shall be recorded.
3.3.2 Exposure set-up. Each device shall be mounted unbiased and have its terminal leads either all shorted or all open.
For MOS devices or any microcircuit containing an MOS element, all leads shall be shorted. An appropriate mounting
fixture which will accommodate both the sample and the required dosimeters (at least one actuation foil and one CaF TLD)
2
shall be used. The configuration of the mounting fixture will depend on the type of reactor facility used and should be
discussed with reactor facility personnel. Test devices shall be mounted such that the total variation of fluence over the
entire sample does not exceed 20 percent. Reactor facility personnel shall determine both the position of the fixture and the
appropriate pulse level or power time product required to achieve the specified neutron fluence level.
METHOD 1017.2
25 August 1983
2

3.4 Exposure. The test devices and dosimeters shall be exposed to the neutron fluence as specified. The exposure level
may be obtained by operating the reactor in either the pulsed or power mode. If multiple exposures are required, the
post-radiation electrical tests shall be performed (see 3.5.1) after each exposure. A new set of dosimeters are required for
each exposure level. Since the effects of neutrons are cumulative, each additional exposure will have to be determined to
give the specified total accumulated fluence. All exposures shall be made at 20°C ±10°C and shall be correlated to a 1 MeV
equivalent fluence.
3.5 Post-exposure.
3.5.1 Electrical tests. Test items shall be removed only after clearance has been obtained from the Health Physicist at
the test facility. The temperature of the sample devices must be maintained at 20°C±10°C from the time of the exposure
until the post-electrical tests are made. The post-exposure electrical tests as specified shall be made within 24 hours after
the completion of the exposure. If the residual radioactivity level is too high for safe handling, this level to be determined by
the local Radiation Safety Officer, the elapsed time before post- test electrical measurements are made may be extended to
1 week. Alternatively, provisions may be made for remote testing. All required data must be recorded for each device after
each exposure.
3.5.2 Anomaly investigation. Parts which exhibit previously defined anomalous behavior (e.g., nonlinear degradation of
.125) shall be subjected to failure analysis in accordance with method 5003, MIL-STD-883.
3.6 Reporting. As a minimum, the report shall include the part type number, serial number, manufacturer, controlling
specification, the date code and other identifying numbers given by the manufacturer. Each data sheet shall include
radiation test date, electrical test conditions, radiation exposure levels, ambient conditions as well as the test data. Where
other than specified electrical test circuits are employed, the parameter measurement circuits shall accompany the data.
Any anomalous incidents during the test shall be fully explained in footnotes to the data.
4. SUMMARY. The following details shall be specified in the request for test or, when applicable, the acquisition
document:
a. Part types.
b. Quantities of each part type to be tested, if other than specified in 3.2.
c. Electrical parameters to be measured in pre and post exposure tests.
d. Criteria for pass, fail, record actions on tested parts.
e. Criteria for anomalous behavior designation.
f. Radiation exposure levels.
g. Test instrument requirements.
h. Radiation dosimetry requirements, if other than 2.3.
i. Ambient temperature, if other than specified herein.
j. Requirements for data reporting and submission, where applicable (see 3.6).
METHOD 1017.2
25 August 1983
3

METHOD 1017.2
25 August 1983
4
