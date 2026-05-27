---
standard: SMC-S-016
method: ""
category: space_environment
language: en
---

# SMC-S-016

# SMC-S-016

SMC Standard SMC-S-016
5 September 2014
------------------------
Supersedes:
SMC-S-016 (2008)
Air Force Space Command
SPACE AND MISSILE SYSTEMS CENTER
STANDARD
TEST REQUIREMENTS
FOR LAUNCH,
UPPER-STAGE AND
SPACE VEHICLES
APPROVED FOR PUBLIC RELEASE; DISTRIBUTION IS UNLIMITED

Change History
Description of Change Effective Date
• Inserted clarifications in Chapter 4, General Requirements June 25, 2014
• Revised Tables 6.3-1, -2, Unit qualification, protoqualification and
acceptance test requirements for units
• Clarified “Evaluation Required” (ER) requirements
• Revised Tables 7.3-1, -2, Subsystem qualification, protoqualification
and acceptance test requirements
• Revised Table 8.3-2, Vehicle acceptance test requirements
• Added text in Chapter 4 General Requirements clarifying requirements
for, and use of, the flightproof test method
• Added criteria for waiver of proof testing on composite structures,
Chapter 6
• Added random vibration reduction of minimum workmanship
guidelines for units weighing >50 lb, Appendix B
• Clarified criteria for unit shock testing, Chapter 6
• Reduced acoustic test spectrum minimums to reflect EELV flight
experience, Appendix B
• Decreased unit thermal cycling requirement on cycles from 27 to 20,
Chapter 6
• Added Option 1, qualification using engineering qualification models
(EQMs)
• Added Option 2, qualification/protoqualification by similarity (QBS)
• Added Option 3, alternate test sequence for vehicle level tests
• Added Option 4, modification of unit level random vibration
protoqualification margins and durations
• Added Option 5, vehicle level flightproof acoustic testing
• Added Option 6, Conditions for deletion of the vehicle level acoustic
test
• Added Option 7, two phase acoustic/vibration
qualification/protoqualification
• Added Option 8, Unit thermal cycle acceptance credit for pre-
conditioning at the board/slice level
• Added Option 9, two tier unit thermal testing
• Added allowable notching methodology for vibration tests
i

| Description of Change | Effective Date |
| --- | --- |
| • Inserted clarifications in Chapter 4, General Requirements • Revised Tables 6.3-1, -2, Unit qualification, protoqualification and acceptance test requirements for units • Clarified “Evaluation Required” (ER) requirements • Revised Tables 7.3-1, -2, Subsystem qualification, protoqualification and acceptance test requirements • Revised Table 8.3-2, Vehicle acceptance test requirements • Added text in Chapter 4 General Requirements clarifying requirements for, and use of, the flightproof test method • Added criteria for waiver of proof testing on composite structures, Chapter 6 • Added random vibration reduction of minimum workmanship guidelines for units weighing >50 lb, Appendix B • Clarified criteria for unit shock testing, Chapter 6 • Reduced acoustic test spectrum minimums to reflect EELV flight experience, Appendix B • Decreased unit thermal cycling requirement on cycles from 27 to 20, Chapter 6 • Added Option 1, qualification using engineering qualification models (EQMs) • Added Option 2, qualification/protoqualification by similarity (QBS) • Added Option 3, alternate test sequence for vehicle level tests • Added Option 4, modification of unit level random vibration protoqualification margins and durations • Added Option 5, vehicle level flightproof acoustic testing • Added Option 6, Conditions for deletion of the vehicle level acoustic test • Added Option 7, two phase acoustic/vibration qualification/protoqualification • Added Option 8, Unit thermal cycle acceptance credit for pre- conditioning at the board/slice level • Added Option 9, two tier unit thermal testing • Added allowable notching methodology for vibration tests | June 25, 2014 |

Contents
1. Scope ...................................................................................................................................... 1
1.1 Purpose ......................................................................................................................... 1
1.2 Application.................................................................................................................... 1
1.3 Baseline Requirements ................................................................................................. 1
1.4 Tailoring........................................................................................................................ 1
1.5 Test Categories ............................................................................................................. 1
1.6 Exclusions for Additional Environments ...................................................................... 2
2. Reference Documents ........................................................................................................... 3
2.1 Applicable Documents .................................................................................................. 3
2.2 Guidance Documents .................................................................................................... 4
3. Definitions ......................................................................................................................... 7
3.1 Assembly ...................................................................................................................... 7
3.2 Ambient Environment ................................................................................................... 7
3.3 Battery ........................................................................................................................... 7
3.4 Burst Factor................................................................................................................... 7
3.5 Computer Program ........................................................................................................ 7
3.6 Design Burst Pressure ................................................................................................... 7
3.7 Design Factor of Safety ................................................................................................ 7
3.8 Design Ultimate Load ................................................................................................... 7
3.9 Design Yield Load ........................................................................................................ 8
3.10 Development Test Article ............................................................................................. 8
3.11 Electromagnetic Compatibility (EMC) Margin ............................................................ 8
3.12 Effective Duration for Acoustics and Random Vibration ............................................. 8
3.13 Explosive Ordnance Device .......................................................................................... 8
3.14 Firmware ....................................................................................................................... 8
3.15 Flight Vehicle ............................................................................................................... 8
3.16 Flight Critical Item ........................................................................................................ 9
ii

3.17 Functional Testing ........................................................................................................ 9
3.18 Hot Operational Soak ................................................................................................... 9
3.19 Launch System ............................................................................................................. 9
3.20 Launch Vehicle............................................................................................................. 9
3.21 Limit Load .................................................................................................................... 9
3.22 Maximum and Minimum Model Temperature Predictions .......................................... 9
3.23 Maximum and Minimum Predicted Temperatures ....................................................... 9
3.24 Maximum Expected Operating Pressure (MEOP) ....................................................... 10
3.25 Maximum Predicted Acceleration ................................................................................ 10
3.26 Maximum Predicted Environment (MPE) for Acoustics ............................................. 10
3.27 Maximum Predicted Environment (MPE) for Random Vibration ............................... 10
3.28 Maximum Predicted Environment (MPE) for Shock ................................................... 10
3.29 Maximum Predicted Environment (MPE) for Sinusoidal Vibration ............................ 11
3.30 Moving Mechanical Assembly (MMA) ....................................................................... 11
3.31 Multipaction ................................................................................................................. 11
3.32 Multi-Unit Module (MUM) .......................................................................................... 11
3.33 On-Orbit System........................................................................................................... 11
3.34 Operational Modes ....................................................................................................... 11
3.35 Part ............................................................................................................................... 12
3.36 Performance Testing ..................................................................................................... 12
3.37 Pressure Component ..................................................................................................... 12
3.38 Pressure Vessel ............................................................................................................. 12
3.39 Pressurized Structure .................................................................................................... 12
3.40 Pressurized Subsystem ................................................................................................. 12
3.41 Proof Factor .................................................................................................................. 13
3.42 Proof Test ..................................................................................................................... 13
3.43 Reusable Item ............................................................................................................... 13
3.44 Service Life .................................................................................................................. 13
3.45 Significant Shock Event ............................................................................................... 13
iii

3.46 Simulator ....................................................................................................................... 13
3.47 Software ........................................................................................................................ 13
3.48 Software Item ................................................................................................................ 13
3.49 Software Unit ................................................................................................................ 13
3.50 Space Vehicle ............................................................................................................... 14
3.51 Statistical Estimates of Vibration, Acoustic, and Shock Environments ....................... 14
3.52 Structural Component ................................................................................................... 14
3.53 Subassembly ................................................................................................................. 14
3.54 Subsystem ..................................................................................................................... 14
3.55 Survival Temperatures .................................................................................................. 14
3.56 System ........................................................................................................................... 15
3.57 Temperature Stabilization ............................................................................................. 15
3.58 Test Discrepancy ........................................................................................................... 15
3.59 Test Item Failure ........................................................................................................... 15
3.60 Thermal Dwell .............................................................................................................. 15
3.61 Thermal Soak ................................................................................................................ 15
3.62 Thermal Uncertainty Margin ........................................................................................ 15
3.63 Unit ............................................................................................................................... 15
3.64 Upper-Stage Vehicle ..................................................................................................... 16
4. General Requirements .............................................................................................................. 17
4.1 Baseline Requirements ................................................................................................. 17
4.1.1 Program Requirements .................................................................................... 17
4.2 Testing Philosophy ....................................................................................................... 17
4.2.1 Development Tests .......................................................................................... 19
4.2.2 Qualification Tests ........................................................................................... 19
4.2.3 Protoqualification Tests ................................................................................... 20
4.2.4 Acceptance Tests ............................................................................................. 20
4.3 Testing Approach .......................................................................................................... 20
4.3.1 Development .................................................................................................... 20
iv

4.3.2 Qualification .................................................................................................... 21
4.3.3 Protoqualification ............................................................................................ 22
4.3.4 Acceptance ...................................................................................................... 24
4.4 Special Considerations ................................................................................................. 24
4.4.1 Propulsion Equipment Tests ............................................................................ 24
4.4.2 Thermal Uncertainty Margins ......................................................................... 25
4.5 Software and Firmware Tests ....................................................................................... 27
4.5.1 Software Development Tests ........................................................................... 27
4.5.2 Software Qualification Tests ........................................................................... 27
4.5.3 Testing of Commercial Off-the-Shelf (COTS) and Reuse Software ............... 27
4.5.4 Software Regression Testing ........................................................................... 27
4.5.5 Other Software-Related Testing ...................................................................... 27
4.6 Inspections .................................................................................................................... 27
4.6.1 Post-Qualification Test Inspections ................................................................. 28
4.6.2 Post-Test Flight Hardware Inspection (Including Launch Site) ...................... 28
4.7 Test Input Tolerances ................................................................................................... 28
4.8 Test Plans and Procedures ............................................................................................ 29
4.8.1 Test Plans ........................................................................................................ 29
4.8.2 Test Procedures ............................................................................................... 30
4.9 Documentation ............................................................................................................. 30
4.9.1 Test Documentation Files ................................................................................ 30
4.9.2 General Test Data ............................................................................................ 31
4.9.3 Qualification, Protoqualification, and Acceptance .......................................... 31
4.9.4 Test Log ........................................................................................................... 31
4.9.5 Test Discrepancy ............................................................................................. 31
4.10 Exceptions to General Requirements ........................................................................... 31
4.10.1 Qualification and Protoqualification by Similarity (QBS) .............................. 31
4.10.2 Electrical and Electronic Unit Thermal Vacuum Test Exemptions ................ 32
v

5. Alternative Strategies ........................................................................................................... 35
5.1 Spares Strategy ............................................................................................................. 35
5.2 Flightproof Strategy ...................................................................................................... 35
5.3 Combination Test Strategies ......................................................................................... 36
5.4 Qualification Using Engineering Qualification Models (EQM) ................................... 36
6. Unit Test Requirements ....................................................................................................... 37
6.1 General Requirements ................................................................................................... 37
6.2 Development Tests ....................................................................................................... 37
6.2.1 Subassembly Development Tests, In-Process Tests, and Inspections ............. 37
6.2.2 Unit Development Tests .................................................................................. 37
6.2.3 Development Tests for Composite Structures ................................................. 37
6.2.4 Thermal Development Tests ............................................................................ 38
6.2.5 Shock and Vibration Isolator Development Tests ........................................... 38
6.2.6 Battery Development Tests .............................................................................. 39
6.2.7 EMI/EMC Development Tests ........................................................................ 39
6.3 Test Program for Units ................................................................................................. 39
6.3.1 Unit Wear-In Test ............................................................................................ 39
6.3.2 Unit Performance and Functional Test ............................................................ 43
6.3.3 Unit Leakage Test ............................................................................................ 44
6.3.4 Unit Shock Test ............................................................................................... 44
6.3.5 Unit Vibration Test .......................................................................................... 46
6.3.6 Unit Acoustic Test ........................................................................................... 50
6.3.7 Unit Acceleration Test ..................................................................................... 53
6.3.8 Unit Thermal Cycle Test .................................................................................. 54
6.3.9 Unit Thermal Vacuum Test ............................................................................. 63
6.3.10 Unit Climatic Tests .......................................................................................... 67
6.3.11 Unit Static Load Test ....................................................................................... 70
6.3.12 Unit Pressure Test ............................................................................................ 72
6.3.13 Unit Electromagnetic Compatibility (EMC) Test ............................................ 74
vi

6.3.14 Unit Life Test .................................................................................................. 75
7. Subsystem Test Requirements ............................................................................................ 77
7.1 Requirements ................................................................................................................ 77
7.2 Subsystem Development Tests ..................................................................................... 79
7.2.1 Mechanical Fit Development Tests ................................................................. 79
7.2.2 Mode Survey Development Tests ................................................................... 79
7.2.3 Structural Development Tests ......................................................................... 79
7.2.4 Acoustic Development Tests ........................................................................... 79
7.2.5 Shock Development Tests ............................................................................... 79
7.2.6 Payload Thermal Development Tests .............................................................. 80
7.3 Test Program for Subsystems ....................................................................................... 80
7.3.1 Subsystem Specification Performance Test .................................................... 80
7.3.2 Subsystem Static Load Test ............................................................................ 82
7.3.3 Subsystem Pressure Test ..................................................................................... 83
7.3.4 Subsystem Vibration Test ............................................................................... 84
7.3.5 Subsystem Acoustic Test................................................................................. 84
7.3.6 Subsystem Shock Test ..................................................................................... 85
7.3.7 Subsystem Thermal Vacuum Test ................................................................... 85
7.3.8 Subsystem Separation and Deployment Tests ................................................. 86
7.3.9 Subsystem Electromagnetic Compatibility (EMC) Test ................................. 87
7.3.10 Mode Survey Test ........................................................................................... 88
8. Vehicle Test Requirements .................................................................................................. 91
8.1 General Requirements .................................................................................................. 91
8.2 Vehicle Development Tests .......................................................................................... 92
8.2.1 Mechanical Fit Development Tests ................................................................. 93
8.2.2 Mode Survey Development Tests ................................................................... 93
8.2.3 Structural Development Tests ......................................................................... 93
8.2.4 Acoustic Development Tests ........................................................................... 93
8.2.5 Shock Development Tests ............................................................................... 93
vii

8.2.6 Thermal Balance Development Tests .............................................................. 94
8.2.7 Transportation and Handling Development Tests ........................................... 94
8.2.8 Wind Tunnel Development Tests .................................................................... 94
8.3 Test Program for Flight Vehicles .................................................................................. 95
8.3.1 Vehicle Specification Performance and Functional Tests ............................... 96
8.3.2 Vehicle Pressure and Leakage Test ................................................................. 97
8.3.3 Vehicle Electromagnetic Compatibility Test ................................................... 98
8.3.4 Vehicle Shock Test .......................................................................................... 100
8.3.5 Vehicle Acoustic Test ...................................................................................... 101
8.3.6 Vehicle Vibration Test ..................................................................................... 103
8.3.7 Vehicle Thermal Balance Test ......................................................................... 105
8.3.8 Vehicle Thermal Vacuum Test ........................................................................ 107
8.3.9 Mode Survey Test ............................................................................................ 108
9. Prelaunch Validation and Operational Tests ..................................................................... 111
9.1 Prelaunch Validation Tests, General Requirements ..................................................... 111
9.2 Prelaunch Validation Test Flow .................................................................................... 111
9.3 Prelaunch Validation Test Configuration ..................................................................... 112
9.4 Prelaunch Validation Test Descriptions ........................................................................ 112
9.4.1 Performance Tests............................................................................................ 112
9.4.2 Propulsion Subsystem Leakage and Functional Tests ..................................... 113
9.4.3 Launch-critical Ground Support Equipment Tests .......................................... 113
9.4.4 Compatibility Test, On-Orbit System .............................................................. 113
9.5 Follow-On Operational Tests for Space Vehicles ......................................................... 114
9.5.1 Follow-On Operational Tests and Evaluations ................................................ 114
9.5.2 On-Orbit Testing .............................................................................................. 114
9.5.3 Reusable Flight Hardware ............................................................................... 114
Appendix A—Thermal Test Considerations ................................................................................ 117
Appendix B—Dynamics Test Considerations .............................................................................. 121
viii

Figures
4.2-1. Baseline Qualification Test Strategy. ....................................................................... 18
4.2-2. Protoqualification Strategy. ...................................................................................... 18
6.3.5-1 Minimum Random Vibration Spectrum, Unit Acceptance Tests............................. 47
6.3.6-1. Minimum Acoustic Levels, Unit and Vehicle. ......................................................... 50
6.3.8-1. Notional Temperature Profile for First Cycle Testing ............................................. 54
6.3.8-2. Notional Temperature Profile for Intermediate Cycle Testing ................................ 57
6.3.8-3. Notional Temperature Profile for Last Cycle Testing .............................................. 57
6.3.8-4. Unit Test Temperature Ranges and Margins............................................................ 59
6.3.10-1. Humidity Test Time Line ......................................................................................... 68
8.3.7-1. Minimum Random Vibration Spectrum, Vehicle Acceptance Tests ....................... 104
A.1.2. Two-Tier Acceptance Thermal Test Profile............................................................. 118
B.1.1 Qualification Test (+6 dB, 3 Minutes) ..................................................................... 126
B.1.2a Phase I Qualification Test for Equivalent Acceptance Life (0 dB, 32 minutes) ...... 126
B.1.2b Phase II Test for Qualification Margin (+6 dB, 1 minute) ....................................... 127
B.1.3 Protoqualification Test (+3 dB, 2 Minutes) ............................................................. 127
B.1.4a. Phase I Protoqualification Test for Equivalent Acceptance Life (0 dB, 32 minutes) 127
B.1.4b Phase II Test for Protoqualification Margin (+3 dB, 1 minute) ............................... 128
B.2.2-1 Weight Adjusted Minimum Spectrum, Unit Acceptance Test ................................. 131
B.2.2-2 Example PSD Notch ................................................................................................ 131
ix

Tables
4.4-1. Categorization of Passive and Active Thermal Hardware ............................................ 25
4.4-2. Thermal Uncertainty Margins for Passive Cryogenic Hardware ................................. 25
4.7-1 Maximum Allowable Test Tolerances ......................................................................... 28
6.3-1. Unit Qualification and Protoqualification Test Summary ............................................ 40
6.3-2. Unit Acceptance Test Summary ................................................................................... 41
6.3-3. Unit Test Level Margins and Duration ......................................................................... 42
6.3.12-1. Unit Pressure Cycle and Burst Test Requirements ....................................................... 73
6.3.12-2. Minimum Design Burst Pressure Requirements .......................................................... 73
7.3-1. Subsystem Qualification and Protoqualification Test Summary .................................. 77
7.3-2. Subsystem Acceptance Test Summary ......................................................................... 78
7.3-3. Subsystem Test Level Margins and Durations ............................................................. 78
8.3-1. Vehicle Qualification and Protoqualification Test Summary ....................................... 91
8.3-2 Vehicle Acceptance Test Summary .............................................................................. 92
8.3-3 Vehicle Test Level Margins and Duration ................................................................... 92
x

1. Scope
1.1 Purpose
This standard establishes the environmental testing requirements for launch vehicles, upper-stage
vehicles, space vehicles, and their subsystems and units. In addition, a uniform set of definitions of
related terms is established.
1.2 Application
This standard is applicable to the procurement of space system hardware as a compliance document
for the establishment of baseline test requirements. The test requirements herein focus on design
verification and the identification of latent defects to help ensure a high level of confidence in
achieving successful space missions.
1.3 Baseline Requirements
This standard establishes the qualification test strategy as the baseline set of test requirements. This
strategy consists of testing dedicated hardware to qualification levels to verify design, followed by
acceptance testing of flight hardware to screen workmanship defects and demonstrate performance.
1.4 Tailoring
It is intended that these test requirements be tailored to each specific program after considering the
design complexity, design margins, vulnerabilities, technology state of the art, in-process controls,
mission criticality, life cycle cost, number of vehicles involved, prior usage, and acceptable risk.
However, the tailored requirements shall achieve a level of verification consistent with the customer’s
risk posture. During the tailoring process, rationale for each tailored requirement shall be docu-
mented. If the baseline requirements in this standard are not tailored by the contract, they stand as
written.
1.5 Test Categories
The tests discussed herein are categorized and defined as follows:
a. Development Tests. Tests conducted on representative articles to characterize engineering
parameters, gather data, and validate the design approach.
b. Qualification Tests. Tests conducted to demonstrate satisfaction of design requirements
including margin and product robustness for designs that have no demonstrated history. A
full qualification validates the planned acceptance program, in-process stress screens, and
retest environmental stresses resulting from failure and rework. In general, qualification
hardware is not flown. Qualification hardware selected for use as flight hardware shall be
evaluated and refurbished as necessary to show that the integrity of the hardware is preserved
and that adequate margin remains to survive the imposed environments and provide useful
life on orbit. For launch vehicles, life includes ground test, launch, ascent and final maneu-
1

vers. Life for on-orbit systems includes ground test, launch, and ascent followed by on-orbit
life.
c. Protoqualification Tests. Tests conducted to demonstrate satisfaction of design requirements
using reduced amplitude and duration margins. These types of tests are generally selected for
designs having limited production where test units will be used for flight. The protoqualifi-
cation test program is supplemented with analyses as well as development and other tests to
demonstrate margin and life for flight of the protoqualification test hardware. Protoqualifica-
tion tests shall validate the planned acceptance program.
d. Acceptance Tests. Vehicle, subsystem, and unit tests conducted to detect workmanship
defects, demonstrate specified functional and performance requirements, and that the hard-
ware is acceptable for delivery.
e. Prelaunch Validation Tests. Prelaunch validation tests conducted at the launch base to
demonstrate readiness of the hardware, software, personnel procedures, and mission inter-
faces to support launch and the program mission.
f. Post-launch Validation Tests. Tests performed following launch to demonstrate perfor-
mance, interface compatibility, calibration, and the ability to meet mission requirements.
1.6 Exclusions for Additional Environments
Environments other than those specified in this standard can be sufficiently stressful as to warrant
special analysis and testing. These include environments such as nuclear and electromagnetic radia-
tion, natural space environment, and lightning.
2

2. Reference Documents
2.1 Applicable Documents
The following documents of the issue in effect on the date of invitation for bids or request for
proposal form a part of this standard to the extent referenced herein.
1. MIL-STD-810G Environmental Engineering Considerations and Laboratory
Tests
2. MIL-STD-1833 (USAF) Test Requirements for Ground Equipment and Associated
Computer Software Supporting Space Vehicles (SMC-S-
024, Test Requirements for Ground Systems)
3. AFSPCMAN 91-710 Range Safety User Requirements Manual
4. AIAA S-080-1998 Metallic Pressure Vessels, Pressurized
Structures, and Pressure Components
5. AIAA S-081A-2006 Composite Overwrapped Pressure Vessels (COPVs)
6. AIAA S-110-2005 Structures, Structural Components, and
Structural Assemblies
7. AIAA S-114-2005 Moving Mechanical Assemblies for Space and Launch
Vehicles
8. AIAA S-113-2005 Criteria for Explosive Systems and Devices on Space and
Launch Vehicles
9. AIAA S-111-2005 Qualification and Quality Requirements for Space Solar
Cells
10. AIAA S-112-2005 Qualification and Quality Requirements for Space Solar
Panels
11. TOR-2003(8583)-2894 Space Systems-Structures Design and Test Requirements
12. TOR-2007(8583)-6889 Reliability Program for Space Systems (SMC-S-013,
Reliability Program for Space Systems)
13. TOR-2005(8583)-3859 Quality Assurance Requirements for Space and Launch
Vehicles (SMC-S-003, Quality Assurance for Launch
Vehicles)
14. TOR-2013(3909)-1 Objective Reuse of Heritage Products
3

15. TOR-2010(8591-20 Flight Unit Qualification Guidelines
16. TOR-2011(8591)-2 V1 Space Vehicle Test and Evaluation Handbook, 2nd
Edition
17. TOR-2003(8583)-2886 Independent Structural Load Analyses of Integrated
Spacecraft/Launch Vehicle Systems (SMC-S-004,
Independent Structural Loads Analysis)
18. TOR-2003(8583)-2896 Space Systems-Flight Pressurized Systems (SMC-S-005,
Space Systems-Flight Pressurized Systems)
19. TOR-2003(8583)-2895, Rev 1 Solid Rocket Motor Case Design and Test Requirements
(SMC-S-006, Solid Rocket Motor Case Design and Test
Requirements)
20. TOR-2004(8583)-5, Rev 1 Space Battery Standard (SMC-S-007, Space Battery)
21. TOR-2005(8583)-1 Electromagnetic Compatibility Requirements for Space
Equipment and Systems (SMC-S-008, Electromagnetic
Compatibility Requirements for Space Equipment and
Systems
22. TOR-2006(8583)-5235, Rev A Parts, Materials, and Processes Control Program for
Space and Launch Vehicles (SMC-S-009, Parts,
Materials, and Processes Control Program for Space and
Launch Vehicles)
23. TOR-2006(8583)-5236 Technical Requirements for Electronic Parts, Materials,
and Processes Used in Space and Launch Vehicles
(SMC-S-010, Parts, Materials, and Processes Used for
Space and Launch Vehicles)
24. TOR-2004(3909)-3537, Rev B Software Development Standard for Space Systems
(SMC-S-012, Software Development for Space Systems)
25. TOR-2008(8583)-8492, Rev A Technical Requirements for Wiring Harness Space
Vehicle, Design and Testing, General Specification
(SMC-S-020, Technical Requirements for Wiring
Harness, Space Vehicle)
26. TOR-2005(8583)-1 Lithium Ion Battery Standards for Spacecraft
Applications (SMC-S-017, Lithium-Ion Battery for
Spacecraft Applications)
27. TOR-2007(8583)-2 Acquisition Standard for Lithium-Ion-Based Launch
Vehicle Batteries (SMC-S-018, Lithium-Ion Battery for
Launch Vehicle Applications)
4

2.2 Guidance Documents
28. MIL-HDBK-340A, Vol. II Test Requirements for Launch, Upper-Stage and Space
Vehicles: Application Guidelines
29. DNA-TR-84-140 Satellite Hardness and Survivability; Testing Rationale
for Electronic Upset and Burnout Effects
30. JANNAF-GL-2012-01-RO Test and Evaluation Guidelines for Liquid Rocket
Engines
31. TOR-2013(3213)-6 Acoustic Testing on Production Space Vehicle (The
Value of the Test and Deletion Conditions)
32. TOR-2006(8506)-4494 Systems Engineering Handbook, Chapter 20,
Manufacturing and Production
Unless otherwise indicated, copies of federal and military specifications, standards, and handbooks
are available from Department of Defense Single Supply Point at http://quicksearch.dla.mil.
AIAA standards must be procured directly from the owner.
Aerospace TORs are available from the Aerospace Corporate Library. Requests, on official letter-
head, should be addressed to:
The Aerospace Corporate Library
Mail Stop M1-199
P.O. Box 92957
Los Angeles, CA 90009-2957
5

THIS PAGE INTENTIONALLY LEFT BLANK

3. Definitions
3.1 Assembly
An integrated set of subassemblies and/or units that comprise a well-defined part of a subsystem.
3.2 Ambient Environment
The ambient environment for a ground test is defined as temperature of 23 ± 3°C (73 ± 5°F), atmo-
spheric pressure of 101 +2/–23 kPa (29.9 +0.6/–6.8 in Hg), and relative humidity of 50 ± 20%.
3.3 Battery
A battery is an assembly of battery cells or modules, electrically connected to provide the desired
voltage and current capability. The battery, encased in a restraining structure, may also include
electrical bypass devices, charge control electronics, heaters, temperature sensors, thermal switches,
and thermal control elements.
3.4 Burst Factor
The burst factor is a multiplying factor applied to the maximum expected operating pressure to obtain
the design burst pressure. Burst factor is synonymous with ultimate pressure factor.
3.5 Computer Program
A computer program is a combination of computer instructions and data that enables computer hard-
ware to perform computational or control functions.
3.6 Design Burst Pressure
The design burst pressure is a test pressure that pressurized components must withstand without rup-
ture in their applicable operating environments
3.7 Design Factor of Safety
The design factor of safety is a multiplying factor used in design analysis to account for uncertainties
such as mechanical tolerances, analysis limitations, and manufacturing variability. The design factor
of safety is often called the design safety factor, factor of safety, or, simply, the safety factor. In gen-
eral, two types of design factors of safety are specified: design yield factor of safety and design ulti-
mate factor of safety.
3.8 Design Ultimate Load
The design ultimate load is a load, or combination of loads, that the structure must withstand without
rupture or collapse in the applicable operating environments. It is equal to the product of the limit
load and the design ultimate factor of safety.
7

3.9 Design Yield Load
The design yield load is a load, or combination of loads, that a structure must withstand without det-
rimental deformation in the applicable operating environments. It is equal to the product of the limit
load and the design yield factor of safety.
3.10 Development Test Article
A development test article is a vehicle, subsystem, or unit dedicated to provide design requirement
information. The information may be used to check the validity of analytic techniques and assumed
design parameters, uncover unexpected response characteristics, evaluate design changes, determine
interface compatibility, demonstrate qualification and acceptance test procedures and techniques, and
to determine whether the equipment meets its performance specifications. Development test articles
are not intended for flight.
3.11 Electromagnetic Compatibility (EMC) Margin
EMC margin is the ratio of the susceptibility of the interface to the emissions present at the interface
from all sources. The EMC margin is to be incorporated into the test levels. Qualification margins of
6 dB are acceptable if the combined test uncertainty, part variation, part degradation at end-of-life,
and workmanship variation are less than 6 dB. Electro-explosive devices and bridge wires have a
20 dB margin requirement below the DC no-fire value and a 6 dB margin requirement below the RF
no-fire value.
3.12 Effective Duration for Acoustics and Random Vibration
To establish basic test requirements, the effective duration in flight for the liftoff and the ascent
acoustic and random environments (max-q and transonic) is taken to be 15 sec to be used in conjunc-
tion with the MPE spectrum (see 3.25 and 3.26). For other sources, the effective duration is the time
within which the overall excitation is within 6 dB of the maximum overall level. Damage-based anal-
ysis methods, such as described in B.1.5, can be used to identify an environment duration and the cor-
responding spectrum.
3.13 Explosive Ordnance Device
An explosive ordnance device is a device that contains, or is operated by, explosives. A cartridge-
actuated device (one type of explosive ordnance device) is a mechanism that employs the energy pro-
duced by an explosive charge to perform or initiate a mechanical action.
3.14 Firmware
Firmware is the combination of a hardware device (including both reprogrammable and non-
reprogrammable devices) and computer instructions and/or computer data that reside as read-only
software on the hardware device.
3.15 Flight Vehicle
The flight vehicle, often referred to as the space segment, is the combined launch system [i.e., the
launch vehicle(s), the upper-stage vehicle(s), and the space vehicle(s)].
8

3.16 Flight Critical Item
A flight-critical item (hardware or software) is one whose failure can affect the system operations suf-
ficiently to cause the loss of the ability to perform the baseline mission or is essential from a range
safety standpoint.
3.17 Functional Testing
Functional testing is performed to assess the operability of the item under test within the boundaries
established by design requirements. For example, the test screens for malfunctions, failure to exe-
cute, sequence of action, interruption in continuous function, or failure in cause and response. Func-
tional tests are conducted in the correct environment. All command functions should be exercised
during functional testing.
3.18 Hot Operational Soak
Hot operational soak is the continuous operating time at hot test temperature on each thermal cycle.
It begins following the hot start on the first and last cycle (Figures 6.3.8-1 and 6.3.8-3) or at the
beginning of the thermal stabilization on intermediate cycles (Figure 6.3.8-2).
3.19 Launch System
A launch system is the composite of elements consisting of equipment, skills, and techniques capable
of launching and boosting one or more space vehicles into orbit. The launch system includes the
flight vehicle and related facilities, ground equipment, material, software, procedures, services, and
personnel required for their operation.
3.20 Launch Vehicle
A launch vehicle is one or more of the lower stages of a flight vehicle capable of launching upper-
stage vehicles and space vehicles, usually into an orbital trajectory. A fairing to protect the space
vehicle during the boost phase is typically considered part of the launch vehicle.
3.21 Limit Load
Limit load is the highest predicted load or combination of loads that a structure or a component in a
structural assembly may experience during its service life in association with the applicable operating
environments. The corresponding stress is called limit stress.
3.22 Maximum and Minimum Model Temperature Predictions
The maximum and minimum model temperature predictions are the hottest and coldest temperatures
predicted from thermal models using applicable effects of worst-case combinations of equipment
operation, internal heating, vehicle orientation, solar radiation, eclipse conditions, ascent heating,
descent heating, and degradation of thermal surfaces during the service life (Figure 6.3.8-4).
3.23 Maximum and Minimum Predicted Temperatures
The maximum and minimum predicted temperatures (MPT) are the highest and lowest temperatures
that an item can experience during its service life, including all test and operational modes. The MPT
are established by adding thermal uncertainty margins to the maximum and minimum model temper-
ature predictions (Figure 6.3.8-4).
9

3.24 Maximum Expected Operating Pressure (MEOP)
The MEOP is the highest pressure that pressurized hardware is expected to sustain during its service
life and retain its functionality. Included are the effects of applicable operating environments, maxi-
mum ullage pressure, fluid head due to vehicle quasi-steady and dynamic accelerations, water ham-
mer, slosh, pressure transients and oscillations, temperature, and operating variability of regulators or
relief valves.
3.25 Maximum Predicted Acceleration
The maximum predicted acceleration, defined for analysis and test purposes, is the highest accelera-
tion determined from the combined effects of quasi-steady acceleration, vibration and acoustics, and
transient flight events (liftoff, engine ignitions and shutdowns, flight through transonic and maximum
dynamic pressure, gust, and vehicle separation). Maximum accelerations are predicted for each of
three mutually perpendicular axes in both positive and negative directions. When a statistical esti-
mate is applicable, the maximum predicted acceleration is at least the acceleration that is not expected
to be exceeded on 99% of flights, estimated with 90% confidence, or P99/90 (B.1.1).
3.26 Maximum Predicted Environment (MPE) for Acoustics
The MPE is statistically the P95/50 acoustic spectrum subject to a constraint discussed in B.1.1. The
acoustic MPE is expressed as a 1/3-octave-band pressure spectrum in dB (Reference 20 µPa) for
center frequencies spanning 31 to 10,000 Hz. For the liftoff and ascent acoustic environments during
a flight, the spectra for each of a series of 1-second durations, overlapped by 50%, are enveloped to
produce the maxi-max flight spectrum. The resulting P95/50 spectrum is 4.9 dB above the log-mean
maxi-max spectrum from a series of flights or tests (B.1.1).
3.27 Maximum Predicted Environment (MPE) for Random Vibration
The MPE is statistically the P95/50 random vibration spectrum, subject to a constraint discussed in
B.1.1. The random vibration MPE is expressed as a spectral density in g2/Hz (commonly, termed the
auto spectral density, ASD, or power spectral density, PSD) calculated at intervals no greater than 1/6
octave over the frequency range of at least 20 to 2000 Hz. For the liftoff and ascent acoustic envi-
ronments during a flight, the spectra for each of a series of 1-second times, overlapped by 50%, are
enveloped to produce the so-called maxi-max flight spectrum. Below 40 Hz, the resolution band-
width need not be less than 5 Hz. The resulting P95/50 spectrum is 4.9 dB above the log-mean maxi-
max spectrum from a series of flights or tests (B.1.1).
3.28 Maximum Predicted Environment (MPE) for Shock
The MPE is statistically the P95/50 shock spectrum, but no less than 4.9 dB above the log mean
spectrum (see B.1.1). The shock MPE is expressed as a shock response spectrum (SRS) in units of
acceleration (g). At each frequency, the shock response spectrum value is the maximum acceleration
response induced by the shock in a single-degree-of-freedom system having a specified natural fre-
quency and amplification, Q. Other methods of characterizing the shock environment, in addition to
the SRS, may be used. Shock transients result from the sudden application or release of loads associ-
ated with deployment, separation, impact, and release events. Such events often employ explosive
ordnance devices, resulting in a so-called pyroshock environment, characterized by a high-frequency
acceleration transient that typically decays within 20 milliseconds. For such transients, the shock
response spectrum is based on a Q of 10 and spans the range from at least 100 Hz to 10,000 Hz at
1 0

intervals no greater than 1/6 octave. If shock isolators are used and have resonances below 100 Hz,
then the range starts below the isolation resonance frequency. For a particular shock event, the
P95/50 shock response spectrum is 4.9 dB above the log-mean shock response spectrum (B.1.1). The
shock MPE is the envelope of the MPE for all shock events.
3.29 Maximum Predicted Environment (MPE) for Sinusoidal Vibration
The sine MPE is the basis for acceptance-level sinusoidal testing. The MPE is statistically the P95/50
sinusoidal vibration spectrum, subject to a constraint discussed in B.1.1. The MPE is expressed as the
amplitude of sinusoidal acceleration, in units of g, over a frequency range of potentially significant
severity as determined by development testing. Typically, a frequency sweep rate in octaves per
minute is specified for a test. The sinusoidal vibration may be due to periodic excitations stemming
from an instability (such as pogo, flutter, combustion) or to those due to rotating machinery.
Significant sinusoidal excitations may also occur during transportation, typically in the frequency
range below 200 Hz.
3.30 Moving Mechanical Assembly (MMA)
A moving mechanical assembly is a mechanical or electromechanical device that controls the move-
ment of one mechanical element of a vehicle relative to another. Examples are gimbals, actuators, de-
spin mechanisms, separation mechanisms, deployment mechanisms, release devices, valves, pumps,
motors, latches, clutches, springs, dampers, and bearings.
3.31 Multipaction
Multipaction is a form of RF voltage breakdown in a vacuum where the electrons impact the elec-
trodes producing more electrons in resonance, resulting in an electrical short.
3.32 Multi-Unit Module (MUM)
A multi-unit module is a testable functional item that is viewed as a complete and separate entity for
purposes of manufacturing, maintenance, and record keeping. Examples: multi-functional box or
module containing boards/slices with a common motherboard or output/input interface. A MUM is
testable as a configured item against its own performance. It contains families of units, slices, or
subassemblies where all of the components may be individually qualified and accepted and meet, at a
minimum, the unit test requirements presented in this document (Tables 6.3-1 and 6.3-2).
3.33 On-Orbit System
The on-orbit system includes the space vehicle(s), the command and control network, and related
facilities, ground equipment, material, software, procedures, services, and personnel required for their
operation.
3.34 Operational Modes
The operational modes for a unit, assembly, subsystem, or system are operational configurations or
conditions that can occur during its service life. Examples include battery charging conditions, com-
mand mode, readout mode, attitude control mode, redundancy management mode, safe mode, and
spinning or despun condition.
1 1

3.35 Part
A part is a single piece, or two or more joined pieces, that is not normally subject to disassembly
without destruction or impairment of the design use. Examples are resistors, integrated circuits,
relays, and roller bearings.
3.36 Performance Testing
Testing conducted to demonstrate measured electrical, optical, and mechanical operation to specifica-
tion requirements before, during, and after satisfying environmental test requirements. Performance
testing demonstrates design margins and specification compliance for all pathways and modes within
the range of requirements.
3.37 Pressure Component
A pressure component is a unit in a pressurized subsystem that is designed primarily to sustain the
acting pressure; excludes pressure vessels, special pressurized equipment (3.38), and pressurized
structure (see 3.39). Examples are propulsion components such as propellant lines and tubes, fittings,
valves, bellows, hoses, regulators, pumps, filters, and accumulators.
3.38 Pressure Vessel
A pressure vessel is a container whose primary purpose is to store pressurized fluids, and has one or
more of the following attributes:
a. Contains stored energy of 19,310 Joules (14,240 ft-lb) or greater, based on adiabatic
expansion of a perfect gas;
b. Contains a gas or liquid that would endanger personnel or equipment or create a mis-
hap if released; or
c. May experience a MEOP greater than 690 kPa (100 psi).
Special pressurized equipment, such as batteries, sealed containers, heat pipes, and cryostats are not
included.
3.39 Pressurized Structure
A pressurized structure is a structure designed to sustain both pressure and vehicle structural loads. A
main propellant tank of a launch vehicle is a typical example.
3.40 Pressurized Subsystem
A pressurized subsystem consists of pressure vessels or pressurized structures, or both, and pressure
components. Electrical or other control units required for subsystem operation are not included.
1 2

3.41 Proof Factor
The proof factor is a multiplying factor applied to the limit load, or maximum expected operating
pressure, to obtain the proof load or proof pressure for use in a proof test.
3.42 Proof Test
A static load or pressure test performed as an acceptance workmanship screen to prove the structural
integrity of a unit or assembly. The proof test gives evidence of satisfactory workmanship and mate-
rial quality by requiring the absence of failure or detrimental deformation. The proof test load and/or
pressure compensates for the difference in material properties between test and design temperature
and humidity, if applicable.
3.43 Reusable Item
A reusable item is a unit, subsystem, or vehicle that is to be used for multiple missions. The service
life of reusable hardware includes all planned reuses, refurbishment, and retesting.
3.44 Service Life
The service life of an item starts at the completion of fabrication and continues through all acceptance
testing, handling, storage, transportation, prelaunch testing, all phases of launch, orbital operations,
disposal, reentry or recovery from orbit, refurbishment, retesting, and reuse that may be required or
specified.
3.45 Significant Shock Event
A significant shock event is one that produces a shock MPE within 6 dB of the envelope of shock
MPEs from all shock events.
3.46 Simulator
A simulator is an electrical, mechanical, or structural unit or part used to validate flight interfaces in
lieu of available flight hardware on one side of the interface.
3.47 Software
Software consists of computer programs and/or data. This includes software residing within firmware
(see 3.14).
3.48 Software Item
A software item is an aggregation of software, such as a computer program or data that satisfies an
end use function. Software items are so designated for purposes of specification, qualification, test-
ing, configuration management, and other purposes.
3.49 Software Unit
A software unit is an element in the design of software, for example, a major subdivision of a soft-
ware item, a component of that subdivision, a class, object, module, function, routine, or data. A
software unit is not the same as a “Unit,” defined in 3.63.
1 3

3.50 Space Vehicle
A space vehicle is an integrated set of subsystems and units, including their software, capable of sup-
porting a specified mission. It can also be called a satellite or spacecraft and includes the integrated
bus and payloads.
3.51 Statistical Estimates of Vibration, Acoustic, and Shock Environments
Qualification and acceptance tests for vibration, acoustic, and shock environments are based upon
statistically expected spectral levels. The level of the extreme expected environment used for qualifi-
cation testing is that level not exceeded on at least 99% of flights, and estimated with 90% confidence
(P99/90 level). The level of the maximum expected environment used for acceptance testing is that
not exceeded on at least 95% of flights, and estimated with 50% confidence (P95/50 level). These
statistical estimates are made assuming a lognormal flight-to-flight variability having a standard devi-
ation of 3 dB, unless a different assumption can be justified. As a result, the P95/50 level estimate is
4.9 dB above the estimated mean (namely, the average of the logarithmic values of the spectral levels
of data from all available flights). When data from N flights are used for the estimate, the P99/90
estimate in dB is 2.0 + 3.9/N1/2 above the P95/50 estimate. When data from only one flight are
available, those data are assumed to represent the mean, and so the P95/50 is 4.9 dB higher and the
P99/90 level is 6 dB higher than the P95/50 level. When ground testing produces the realistic flight
environment (e.g., engine operation or activation of explosive ordnance), the statistical distribution
can be determined using the test data, provided data from a sufficient number of tests are available
(B.1.1).
3.52 Structural Component
A mechanical unit is considered a structural component if it sustains load and/or pressure or maintains
alignment.
3.53 Subassembly
A subassembly is an item containing two or more parts, which is capable of disassembly or part
replacement. Examples: printed circuit board with parts installed, gear train.
3.54 Subsystem
A subsystem is an assembly of functionally related units, including any associated software. It con-
sists of two or more units and may include interconnection items such as cables or tubing, and the
supporting structure to which they are mounted. Examples: structure, electrical power, attitude con-
trol, telemetry, thermal control, and propulsion subsystems, spacecraft payloads, and instruments.
3.55 Survival Temperatures
Survival temperatures are the cold and hot temperatures a unit is expected to survive. Survival tem-
perature limits are specified as operational and/or non-operational. For an operational survival limit,
unit survival is the demonstration that the unit can operate at the survival limit. Although the unit
does not need to meet performance to specification, it cannot show any performance degradation
when the unit is returned to the operational or acceptance temperature range. For a non-operational
survival limit, unit survival is the demonstration that the unit can be taken to the survival limit with
the unit off and show no performance degradation when the unit is returned to the functional or per-
formance temperature range.
1 4

3.56 System
A system is a composite of equipment, skills, and techniques capable of performing or supporting an
operational role. A system includes all operational equipment, related facilities, material, software,
services, and personnel required for its operation.
3.57 Temperature Stabilization
Temperature stabilization in a thermal test is achieved when the controlling temperature location on
the test article is within the allowed test tolerance at the specified test temperature and the tempera-
ture rate of change is less than a specified value.
3.58 Test Discrepancy
A test discrepancy is any anomalous or unexpected condition encountered during a test process. Test
discrepancies include those associated with performance, premature operation, and failure to operate.
3.59 Test Item Failure
A failure of a test item is defined as a test discrepancy that is due to a design, workmanship, process,
or any quality deficiency in the item being tested. Any test discrepancy is considered a failure of the
test item unless it can be determined to have been due to an unrelated cause.
3.60 Thermal Dwell
Thermal dwell of a unit at the hot or cold temperature extreme is the time required to ensure that
internal parts and subassemblies have achieved thermal equilibrium. Thermal dwell begins at the
onset of temperature stabilization and is followed by performance testing.
3.61 Thermal Soak
Thermal soak consists of the total time that a test article is continuously maintained within the
allowed tolerance of the specified test temperature. It begins at the onset of thermal stabilization and
concludes at the end of performance testing.
3.62 Thermal Uncertainty Margin
The thermal uncertainty margin is included in the thermal analysis of units, subsystems, and space
vehicles to account for uncertainties in modeling parameters such as complicated view factors, sur-
face properties, contamination, radiation environments, joint conduction, and inadequate ground sim-
ulation. For units that have only passive thermal control, the thermal uncertainty margin is a temper-
ature added to analytic thermal model predictions. For units with active thermal control, the thermal
uncertainty margin is a control authority (Figure 6.3.8-4).
3.63 Unit
A unit is a functional item (hardware and, if applicable, software) that is viewed as a complete and
separate entity for purposes of manufacturing, maintenance, record keeping and environmental test-
ing. Examples: hydraulic actuator, valve, battery, and transmitter.
1 5

3.64 Upper-Stage Vehicle
An upper-stage vehicle is a vehicle that has one or more stages of a flight vehicle capable of injecting
a space vehicle or vehicles into orbit from the suborbital trajectory.
1 6

4. General Requirements
This section addresses general environmental test requirements for space systems. They consist of
test requirements for units, subsystems, and flight systems and include inspection, test condition tol-
erances, test plans and procedures, retest, and documentation requirements.
4.1 Baseline Requirements
This standard establishes a baseline of requirements as part of a verification program. The baseline
strategy is unit, subsystem, and vehicle-level qualification and acceptance to provide a high level of
probable success for meeting performance requirements over mission life. The specific requirements
applicable at these three levels of assembly are presented in 6, 7, and 8. For those programs that
utilize a different test strategy, the baseline approach provides a benchmark for assessing program
risk. Regardless of the test strategy, hardware shall be designed to qualification levels.
4.1.1 Program Requirements
The requirements presented in this standard are intended to be tailored for program contractual use.
Deviations from the final tailored requirements shall require pre-approval from the customer.
4.2 Testing Philosophy
The complete test program for launch vehicles, upper-stage vehicles, and space vehicles encompasses
development, qualification, acceptance, system, pre-launch validation, and post-launch validation
tests. Test methods, environments, and measured parameters shall be selected to permit the collection
of empirical design or performance data for correlation or trending throughout the test program. See
Reference 28 for further guidance.
A satisfactory test program requires the completion of specific test objectives in a specified sequence.
The test program encompasses the testing of progressively more complex assemblies of hardware and
computer software. Design suitability should be demonstrated in the earlier development tests prior
to formal qualification testing. All qualification testing for an item shall be completed, and design
modifications incorporated prior to the initiation of flight hardware acceptance testing.
The baseline qualification test strategy is shown schematically in Figure 4.2-1. This strategy consists
of designing and testing dedicated hardware to qualification levels to verify the design, demonstrate
margin, and validate an acceptance test program allowing multiple rework cycles. Subsequent flight
hardware shall be acceptance tested to demonstrate functional and/or performance to specification.
The protoqualification test strategy is shown schematically in Figure 4.2-2. This strategy consists of
testing flight hardware to protoqualification levels for limited design verification, demonstration of
reduced margin, functional and/or performance testing, and workmanship screening.
1 7

Figure 4.2-1. Baseline qualification test strategy.
Figure 4.2-2. Protoqualification strategy.
Subsequent flight hardware is acceptance tested. This strategy does not allow multiple retest cycles.
(See B.1.2 for dynamic environments.)
The requirement to design hardware to qualification levels (see 4.1) provides the basis for the pro-
toqualification test strategy. This approach mitigates the risk inherent in flying protoqualification
tested hardware without demonstration of qualification margins. (See B.1.3 for dynamic
environments.)
A brief overview of alternate strategies, such as the flightproof strategy, the judicious use of test
hardware as spares, and combinations of qualification and protoqualification strategies, is presented
in 5.
1 8

The environmental tests specified are intended to be imposed sequentially, rather than in combination.
Nevertheless, features of the hardware design or of the service environments may warrant the imposi-
tion of combined environments in some tests. Examples include combined temperature, acceleration,
and vibration when testing units employing elastomeric isolators in their design; and combined shock,
vibration, and pressure when testing pressurized components. In formulating the test requirements in
these situations, a logical combination of environmental factors should be imposed to enhance test
perceptiveness and effectiveness.
4.2.1 Development Tests
Development tests, or engineering tests, may be performed to:
a. Evaluate new design concepts or the application of proven concepts and techniques
to a new configuration.
b. Assist in the evolution of designs from the conceptual phase to the operational phase.
c. Validate design changes.
d. Reduce the risk involved in committing designs to the fabrication of qualification and
flight hardware.
e. Develop and validate qualification and acceptance test procedures.
f. Investigate problems or concerns that arise after successful qualification.
Requirements for development testing therefore depend upon the maturity of the subsystems and units
used, and upon the operational requirements of the specific program. An objective of development
testing is to identify problems early in their design evolution so that corrective actions can be taken
prior to starting formal qualification testing. Development tests should be used to confirm structural
and performance margins, manufacturability, testability, maintainability, reliability, life expectancy,
and compatibility with system safety. Where practical, development tests should be conducted over a
range of operating conditions that exceeds the design limits to identify marginal capabilities and mar-
ginal design features. Comprehensive development testing is an especially important ingredient to
mission success in programs that plan to use qualification items for flight, including those that allow a
reduction in the qualification test levels and durations. Development tests may be conducted on
breadboard equipment, prototype hardware, or the development test vehicle equipment.
4.2.2 Qualification Tests
Qualification testing (Figure 4.2-1) is conducted to demonstrate that the design, manufacturing pro-
cess, and acceptance program produce hardware/software meeting specification requirements with
adequate margin to accommodate multiple rework and test cycles. In addition, the qualification tests
shall validate the planned acceptance program, including test techniques, procedures, equipment,
instrumentation, and software. A full qualification ensures that subsequent hardware production units
remain flightworthy after surviving multiple acceptance tests that may be necessary because of next
assembly failures and/or overstress that may precipitate rework. A single qualification test specimen
of a given design shall be exposed to all applicable environmental tests. The use of multiple qualifi-
1 9

cation test specimens may be required for one-time-use devices (such as explosive ordnance or solid-
propellant rocket motors).
4.2.3 Protoqualification Tests
Protoqualification testing (Figure 4.2-2) applies reduced amplitude and duration margins to flight
hardware. This testing strategy leads to a higher level of risk, unless mitigated by other testing and
analyses. It also presents reduced retest opportunities in the event of hardware failure, and the poten-
tial for late discovery of design defects.
The protoqualification described in this section may be used at the vehicle, subsystem, and unit lev-
els. Acceptance testing shall be conducted on all subsequent flight items. The protoqualification
strategy shall require technical justification demonstrating that the strategy meets program require-
ments. Alternate strategies are discussed in 5.
4.2.4 Acceptance Tests
Acceptance tests shall be conducted on each deliverable item to demonstrate acceptable quality of
workmanship, functional capability, and performance to specifications. Acceptance testing is intended
to stress screen items to precipitate failures due to latent defects.
If the equipment is to be used by more than one program or in different vehicle locations, the accep-
tance test conditions shall envelop the worst-case environments. For certain items the specified
acceptance test environments could result in physical deterioration of materials or other damage. In
those cases, less severe acceptance test environments that still satisfy the system operational require-
ments may be used.
Acceptance testing as discussed in this section does not apply to hardware acquired through lot
acceptance testing.
4.3 Testing Approach
4.3.1 Development
Development tests on representative unit, subsystem, or vehicle hardware may be performed to evalu-
ate design feasibility, performance acceptability or to obtain engineering data. The development test
plan may use hardware such as breadboards, engineering models, or development models.
4.3.1.1 Parts, Materials, and Process Development Tests and Evaluations
Parts, materials, and process development tests and evaluations are conducted to demonstrate the fea-
sibility of using certain items or processes in the implementation of a design. Development tests,
evaluations, and subsequent qualifications are required for new types of parts, materials, and
processes. References 22 and 23 shall be used as a source of requirements for this process.
2 0

4.3.2 Qualification
4.3.2.1 Qualification Hardware
The hardware subjected to qualification testing shall be produced in the same factory from the same
drawings, using the same parts, materials, tooling, manufacturing process, and level of personnel
competency as used for flight hardware.
A single qualification test specimen of a given design shall be exposed to all applicable environmen-
tal tests. The use of multiple qualification test specimens may be required for one-time-use devices
(such as explosive ordnance devices).
A vehicle or subsystem qualification test article shall be fabricated using qualification units to the
maximum extent practical. Modifications are permitted, if required, to accommodate benign changes
that may be necessary to conduct the test. These changes include adding instrumentation to record
functional parameters, test input and response data, or design parameters for engineering evaluation.
When structural items are rebuilt or reinforced to meet specific strength or rigidity requirements, all
modifications shall be structurally identical to the changes incorporated in flight articles.
The testing allowed prior to the start of qualification testing of an item includes:
a. Wear-in or run-in necessary to achieve a smooth, consistent, and controlled operation
of MMAs,
b. In-process workmanship screening, and
c. Burn-in of certain electrical/electronic assemblies to screen latent defects.
Acceptance testing of qualification hardware may be conducted to verify successful performance at
acceptance levels before proceeding to higher levels for formal qualification testing in that environ-
ment. For those environments that are applied by axis, both the acceptance and qualification tests
may be completed in one axis before switching to another.
4.3.2.2 Qualification Test Levels and Durations
The qualification test level for an environment shall provide a specified margin to the acceptance
level. Qualification test durations or repetitions demonstrate life remaining for flight after a maxi-
mum time or repetitions of acceptance testing at all levels of assembly in support of rework or retest
of flight hardware. Qualification testing should not cause unrealistic modes of failure. If the hard-
ware is to be used by more than one program or in different applications within the same program, the
qualification test conditions should envelop the worst-case application. Required qualification mar-
gins and durations are summarized in the following chapters for unit, subsystem, and system testing.
4.3.2.3 Qualification Retest
Qualification retests occur when the design has changed the form, fit, or function of the hardware or
when the hardware service environment has reduced or eliminated demonstrated qualification mar-
gins. Qualification tests shall be repeated in the impacted environments and include any invalidated
2 1

tests. Re-qualification testing shall also include performance testing and validate all related
interfaces.
Retesting may also be necessary if a test discrepancy or test item failure occurs while performing any
of the required testing steps. After performing a failure analysis, which identifies and isolates the root
cause of the failure, a retest in the failed environment shall be performed. When previous tests have
been invalidated by the failure, those tests shall be repeated. See Reference 16 for further retest
guidance.
The minimum retesting for units and Multi-Unit Modules (MUM) shall consist of three axes of ran-
dom vibration and three thermal cycles or thermal vacuum cycles. The random vibration retest shall
be conducted at qualification levels and durations and the unit or MUM shall be powered on and
monitored. The choice of thermal cycling or thermal vacuum testing shall be consistent with the
thermal testing performed for the baseline qualification of the unit. The thermal cycles shall be con-
ducted at qualification test temperatures and include performance testing at the hot and cold temper-
ature extremes during the first and the last cycle. More extensive rework requires retesting that
repeats the entire acceptance and qualification test sequence. See 4.3.4.3 for testing the reworked
qualification test item to verify workmanship before retesting to qualification levels. Justification
shall be required and approval received if the retest is to be performed at a different level of integra-
tion than that in which the discrepancy was detected.
4.3.3 Protoqualification
The protoqualification strategy requires the design of the hardware to qualification level environ-
ments and testing to protoqualification levels. Subsequent to protoqualification testing, a baseline
acceptance program shall be conducted on all other flight items.
The protoqualification strategy applies when test hardware is used for flight. This strategy introduces
a higher level of risk into a program since it does not establish service life for the first flight hard-
ware, unless mitigated by other testing. Limited life is established for subsequent flight hardware.
Pressurized vessels and components cannot follow a protoqualification approach in regard to pressure
testing. Verification of pressure capability must, instead, conform to a qualification approach
although other requirements could then follow a protoqualification strategy (see 6.3.12).
4.3.3.1 Protoqualification Hardware
The hardware subjected to protoqualification testing shall be produced from the same drawings, using
the same materials, tooling, manufacturing process, and level of personnel competency as acceptance
hardware.
A single protoqualification test specimen of a given design shall be exposed to all applicable envi-
ronmental tests. When practical, the protoqualification test specimen shall be selected randomly from
a group of production items. The use of multiple protoqualification test specimens is required for
one-time-use devices (such as explosive ordnance or solid-propellant rocket motors).
2 2

A vehicle or subsystem protoqualification test article shall be fabricated using protoqualification units
to the maximum extent practical. Modifications are permitted if required to accommodate benign
changes that may be necessary to conduct the test. These changes include adding instrumentation to
record functional parameters, test input and response data, or design parameters for engineering eval-
uation. The only tests allowed prior to the start of protoqualification testing of an item include:
a. Wear-in or run-in necessary to achieve a smooth, consistent, and controlled operation
of MMAs
b. In-process workmanship screening, and
c. Burn-in of certain electrical/electronic assemblies to screen out latent defects
4.3.3.2 Protoqualification Test Levels and Durations
The protoqualification test level for an environment shall include margin and duration, reduced from
that for qualification. The protoqualification test demonstrates that subsequent hardware has life
remaining for flight after acceptance testing at all levels of assembly, but demonstrates little or no
margin for retest (see B.1.2). Protoqualification testing should not create conditions that exceed
applicable safety margins or cause unrealistic modes of failure. If the equipment is to be used by
more than one program or in different applications within the same program, the test conditions
should envelop the worst-case application. Required margins and durations are summarized in the
following chapters for unit, subsystem, and system testing.
4.3.3.3 Protoqualification Retest
Protoqualification retesting is required when the design has changed the form, fit, or function of the
hardware or when predicted environments have increased. Protoqualification retesting shall be per-
formed at protoqualification levels and durations. Protoqualification retesting shall include perfor-
mance testing and validation of all related interfaces.
Retesting may also be necessary if a test discrepancy or test item failure occurs while performing any
of the required testing steps. Following the identification of the root cause and necessary hardware
rework, a retest in the failed environment shall be performed. Customer approval shall be required if
the retest is to be performed at a different level of integration than that in which the discrepancy was
detected. When previous tests have been invalidated by the failure, those tests shall be repeated.
The minimum retesting for units and Multi-Unit Modules (MUM) shall consist of three axes of ran-
dom vibration and three thermal cycles or thermal vacuum cycles. The random vibration retest shall
be conducted at protoqualification levels and durations and the unit or MUM shall be powered on and
monitored. The choice of thermal cycling or thermal vacuum testing shall be consistent with the
thermal testing performed for the baseline protoqualification of the unit. The thermal test shall be
conducted at protoqualification test temperatures and include performance testing at the hot and cold
temperature extremes during the first and the last cycle. More extensive rework requires retesting
that repeats the entire protoqualification test sequence. See 4.3.4.3 for workmanship screening to be
performed prior to retesting to protoqualification levels. Adequate life for retesting and flight shall be
established. Justification shall be required and customer approval received if the retest is to be per-
formed at a different level of integration than that in which the discrepancy was detected.
2 3

4.3.4 Acceptance
4.3.4.1 Acceptance Hardware
The item subjected to acceptance testing shall be flight hardware, including software, as applicable.
4.3.4.2 Acceptance Test Levels and Durations
To demonstrate workmanship, the acceptance environmental conditions shall stress the hardware to
the maximum conditions expected for all flight events, including transportation and handling, but not
less than the minimum workmanship levels defined by this standard. Required margins on flight and
acceptance test levels and durations are summarized in the following chapters for unit, subsystem,
and system testing.
4.3.4.3 Acceptance Retest
Retesting shall be necessary if a test discrepancy or test item failure occurs while performing any of
the required testing steps (see 3.58). After performing a failure analysis, which identifies, isolates,
and corrects the root cause of the failure, a retest in the failed environment shall be performed. Justi-
fication shall be required and approval received if the retest is to be performed at a different level of
integration than when the discrepancy was detected. Customer approval shall be required if the retest
is to be performed at a different level of integration than that in which the discrepancy was detected.
When previous tests have been invalidated by the failure, those tests shall be repeated.
To verify workmanship after rework, the minimum retesting for units and Multi-Unit Modules
(MUM) shall consist of three axes of random vibration and three thermal cycles or thermal vacuum
cycles. The random vibration retest shall be conducted at acceptance levels for one minute per axis
and the unit or MUM shall be powered on and monitored. The choice of thermal cycling or thermal
vacuum testing shall be consistent with the thermal testing performed for the baseline acceptance of
the unit. The thermal cycles shall be conducted at acceptance test temperatures and include perfor-
mance testing at the hot and cold temperature extremes during the first and the last cycle. More
extensive rework requires retesting that repeats the entire acceptance test sequence. Justification shall
be required and approval received if the retest is to be performed at a different level of integration
than that in which the discrepancy was detected.
4.4 Special Considerations
4.4.1 Propulsion Equipment Tests
Units that make up a vehicle propulsion subsystem, including units that are integral to or mounted on
a motor or engine are covered by this Standard in that they shall be qualified and acceptance tested to
the applicable unit requirements specified herein. Testing of a unit on an engine during the engine
acceptance test firing may be substituted for part of the unit level acceptance test if it can be estab-
lished that the environments and duration meet the intent of the individual acceptance test criteria, or
if such units are not amenable to testing individually. Environmental testing of thrusters (such as
staging rockets, retro-motors, and attitude control thrusters) shall meet the applicable unit require-
ments of this Standard. Further guidance on functional testing of launch vehicle engines is addressed
in Reference 30.
2 4

4.4.1.1 Engine Line Replaceable Unit (LRU) Acceptance Testing
An engine LRU is a unit that may be removed and replaced by a new unit, without requiring re-
acceptance test firing of the engine with the new unit. If the unit being replaced was included in an
engine acceptance test firing as part of its acceptance test, then the replacement unit shall either be
subjected to such a test on an engine, or shall undergo equivalent unit-level acceptance testing.
Equivalent testing shall consider all appropriate environments, such as temperature, vibration, pres-
sure, vacuum, and chemical. Testing shall demonstrate functionality and performance of the unit
under conditions similar to those achieved in the engine acceptance test firing and flight.
4.4.1.2 Engine Line Replaceable Unit (LRU) Qualification Testing
All engine LRUs shall be qualified at a unit level to the requirements of this Standard.
4.4.2 Thermal Uncertainty Margins
For the purpose of thermal uncertainty margin specification, thermal control hardware is categorized
as either passive or active. Passive hardware uses a thermal uncertainty margin, whereas active
hardware uses excess power as a thermal uncertainty margin. Examples of passive and active thermal
control hardware for purposes of uncertainty margin are identified in Table 4.4-1.
Table 4.4-1. Categorization of Passive and Active Thermal Hardware
Passive Active
• Constant conductance or diode heat pipes • Variable conductance heat pipes
• Hardwired heaters (fixed or variable resistance, • Heat pumps and refrigerators
such as auto trace or positive temperature coef-
ficient thermistors)
• Thermal storage devices (such as phase- • Stored coolant subsystems
change or sensible heat)
• Thermal insulator (such as multilayer insulation, • Resistance heater with commandable or
foams, or discrete shields) mechanical or electronic controller (including
proportional control)
• Radiators (fixed articulated or deployable) with • Capillary-pumped loops and loop heat pipes
louvers or pinwheels
• Surface finishes (such as coating, paints, • Pumped fluid loops
treatments, second-surface mirrors)
• Thermoelectric cooler
4.4.2.1 Margins for Passive Thermal Control Hardware
For units that have only passive thermal control, the minimum thermal uncertainty margin shall be
11°C. For units that have large uncertainties in operational or environmental conditions, the thermal
uncertainty margin may be greater than 11°C. Examples of these units for a launch vehicle are a
vehicle heat shield, external insulation, and units within the aft skirt. For any unit or subsystem that
is not thermal balance tested, the thermal uncertainty margin shall be 17°C.
a. Margin for Radiators. Radiator margins shall be implemented based upon analytic predic-
tions using worst-case power modes and environments. Prior to thermal model correlation
with thermal balance test data, the design of radiators shall include 10% excess radiator area
2 5

in addition to normal power growth uncertainties. When technically justified and approved
by the customer, the 10% margin may be reduced for the final mission phase predictions
made with the correlated thermal model.
b. Margin for Passive Hardware at Cold Temperatures. Uncertainty margins for spacecraft
hardware operating below -70°C (including passive cryogenic subsystems) may be reduced
as presented in Table 4.4-2.
Table 4.4-2. Thermal Uncertainty Margins for Passive Hardware at Cold Temperatures
Predicted Temperature Thermal Uncertainty Margin
(°C) (°C)
Above –70 11
–70 to –87 10
–88 to –105 9
–106 to –123 8
–124 to –141 7
–142 to –159 6
–160 to –177 5
–178 to –195 4
–196 to –213 3
–214 to –232 2
Below –232 1
4.4.2.2 Margins for Active Thermal Control Hardware
Thermal designs in which temperatures are actively controlled shall use a power margin of 25%
above the minimum design value in lieu of the thermal margins specified in 4.4.2.1. This margin is
applicable at the condition that imposes the maximum or minimum expected temperatures. For
example, for heaters regulated by a mechanical thermostat or electronic controller, a 25% heater
capacity margin may be used in lieu of the thermal margins at the minimum expected temperature and
at minimum bus voltage, which translates into a duty cycle of no more than 80% under these cold
conditions.
4.4.2.2.1 Margins for Cryogenic Hardware
Subsystem designs in which the temperatures are actively controlled to below –70°C by expendable
coolants or refrigerators shall have a thermal uncertainty heat-load margin of 50% for the conceptual
phase, 45% for the preliminary design review phases, 35% for the critical design review phase, 30%
for the qualification hardware, and 25% for the acceptance hardware.
4.4.2.3 Margins for Units Controlled by Heat Pipes
All units whose temperatures are controlled by heat pipes (constant conductance or variable conduc-
tance) shall demonstrate that maximum model temperature predictions of the units can be maintained
within the unit’s acceptance temperature limits should any one of the controlling heat pipes fail.
Demonstration is by analysis, wherein conductive heat paths to heat pipes are individually discon-
nected and resulting temperature predictions are compared to acceptance limits. A minimum 25%
excess heat transport margin shall be maintained by the remaining operating heat pipes. These two
provisions constitute the requirement for heat pipe redundancy.
2 6

4.4.3 Explosively Actuated Devices
For explosively actuated devices, the requirements specified in Reference 8 shall be satisfied.
4.5 Software and Firmware Tests
4.5.1 Software Development Tests
Software development testing verifies that the software performs as designed. Software development
testing shall include software unit testing, software unit integration testing, and software/hardware
integration testing.
Software unit testing shall be performed in accordance with Reference 24.
Software unit integration testing shall be performed in accordance with Reference 24. Soft-
ware/hardware integration testing shall be performed in accordance with Reference 24.
4.5.2 Software Qualification Tests
Software qualification testing verifies that the software meets its specified requirements. Software
qualification testing shall be performed in accordance with Reference 24, paragraph 5.9 and its
subparagraphs.
4.5.3 Testing of Commercial Off-the-Shelf (COTS) and Reuse Software
Testing of COTS and reuse of software shall be performed in accordance with Reference 24, para-
graphs, 5.7.2, 5.8.1, 5.9.3, and 5.10.1.
4.5.4 Software Regression Testing
Regression testing of affected software unit test cases, software unit integration test cases, soft-
ware/hardware integration test cases, and software qualification test cases shall be performed after
any modification to previously tested software. Regression testing of appropriate software unit inte-
gration, software/hardware integration and/or software qualification test cases shall be performed
after the initial loading of the operational flight constants and also after loading any changes to the
operational flight constants.
4.5.5 Other Software-Related Testing
Software shall be included along with hardware in all types of testing specified by this standard where
the software is needed in order to verify the functionality and performance of the hardware itself or of
the integrated hardware/software unit, subsystem, or system.
4.6 Inspections
All units and higher levels of assembly shall be inspected to identify discrepancies before and after
testing. Disassembly or removal of unit covers during inspections shall only be allowed when spe-
cifically called out in the test procedures. Included should be applicable checks of finish, identifica-
tion markings, and cleanliness. Weight, dimensions, clearances, fastener tightness torques, and
2 7

breakaway forces and torques shall be measured, as applicable, to determine compliance with specifi-
cations. (See Reference 13.)
Upon completion of the environmental test program, tested hardware shall be inspected as follows:
4.6.1 Post-Qualification Test Inspections
Inspection of unit/subsystem hardware following completion of qualification shall entail disassembly
to the extent that wear and/or mechanical integrity can be confirmed (e.g., fractures in circuit boards
are not present, heavy component staking is in place, there are no broken brackets, wedge locks and
internal connectors are secure, etc.). Moving mechanical assemblies that undergo life test can be
subjected to an abbreviated inspection sufficient to confirm viability to continue on to the life test
followed by a complete disassembly inspection at the conclusion of life testing.
4.6.2 Post-Test Flight Hardware Inspection (Including Launch Site)
Flight hardware shall be inspected following environmental testing. Inspection should include appli-
cable checks of finish, identification markings, and cleanliness. Weight, dimensions, clearances, fas-
tener tightness torques, and breakaway forces and torques shall be measured, as applicable, to deter-
mine compliance with specifications. Inspection of flight hardware shall not entail the removal of
unit covers or any specific disassembly unless called out in the test procedures.
4.7 Test Input Tolerances
Unless stated otherwise, the specified test parameters shall include the maximum allowable test toler-
ances listed in Table 4.7-1.
Table 4.7-1 Maximum Allowable Test Tolerances
Test Parameters Test Tolerance
Temperature
–54°C to +100°C ± 3°C
Relative Humidity ± 5%
Acceleration +10/–0%
Static Load and Pressure +5/–0%
Atmospheric Pressure
Above 133 Pa (>1 Torr) ±10%
133 to 0.133 Pa (1 Torr to 0.001 Torr) +10/–25%
Below 0.133 Pa (<0.001 Torr) +0/–80%
Test Time Duration +10/–0%
Vibration Frequency ± 2%
Random Vibration Power (Auto) Spectral Density (g2/Hz)
Frequency Range Maximum Control Bandwidth
20 to 100 Hz 10 Hz ± 1.5 dB
100 to 1000 Hz 10% of mid-band frequency ± 1.5 dB
1000 to 2000 Hz 100 Hz ± 3.0 dB
Overall Level (Grms) ± 1.0 dB
Note: Control bandwidths may be combined for tolerance
evaluation purposes. The statistical degrees of freedom shall
be at least 100.
Sound Pressure Levels
1/3-Octave Midband Frequencies
31.5 to 40 Hz ± 5.0 dB
2 8

| Test Parameters | Test Tolerance |
| --- | --- |
| Temperature –54°C to +100°C | ± 3°C |
| Relative Humidity | ± 5% |
| Acceleration | +10/–0% |
| Static Load and Pressure | +5/–0% |
| Atmospheric Pressure Above 133 Pa (>1 Torr) 133 to 0.133 Pa (1 Torr to 0.001 Torr) Below 0.133 Pa (<0.001 Torr) | ±10% +10/–25% +0/–80% |
| Test Time Duration | +10/–0% |
| Vibration Frequency | ± 2% |
| Random Vibration Power (Auto) Spectral Density (g2/Hz) Frequency Range Maximum Control Bandwidth 20 to 100 Hz 10 Hz 100 to 1000 Hz 10% of mid-band frequency 1000 to 2000 Hz 100 Hz Overall Level (Grms) Note: Control bandwidths may be combined for tolerance evaluation purposes. The statistical degrees of freedom shall be at least 100. | ± 1.5 dB ± 1.5 dB ± 3.0 dB ± 1.0 dB |
| Sound Pressure Levels 1/3-Octave Midband Frequencies 31.5 to 40 Hz | ± 5.0 dB |

Test Parameters Test Tolerance
50 to 2000 Hz ± 3.0 dB
2500 to 10000 Hz ± 5.0 dB
Overall SPL ± 1.5 dB
Note: The statistical degrees of freedom shall be at least 100.
Shock Response Spectrum (Peak Absolute Acceleration, Q = 10)
Natural Frequencies Spaced at 1/6-Octave Intervals
At or below 3000 Hz ± 6.0 dB
Above 3000 Hz + 9.0/–6.0 dB
Note: At least 50% of the spectrum values shall be greater than
the nominal test specification.
Electromagnetic Compatibility ± 2 dB
4.8 Test Plans and Procedures
The test plans and procedures shall be documented in sufficient detail to provide a framework for
identifying and interrelating all of the individual tests and test procedures needed.
4.8.1 Test Plans
The test plans shall provide a general description of each test planned and the conditions of the tests.
The test plans shall be based upon a function-by-function mission analysis and any specified testing
requirements. To the degree practical, tests shall be planned and executed to fulfill test objectives
from development through operations. Test objectives shall be planned to verify compliance with the
design and specified requirements of the items involved, including interfaces.
Test plans shall include an allowable experimental uncertainty requirement for each measured param-
eter. Each allowable uncertainty statement shall include a positive and negative uncertainty. Such
experimental uncertainty requirements shall support the objectives of the test.
As a minimum, the test plan shall address the following:
a. The allocation of requirements to appropriate testable levels of assembly. Usually
this is a reference to a requirements traceability matrix listing all design requirements
and indicating a cross-reference to a verification method and to the applicable assem-
bly level.
b. The identification of separate environmental test zones (such as the engine, fairing, or
payload)
c. The identification of separate states or modes where the configuration or environ-
mental levels may be different (such as during testing, launch, upper-stage transfer,
on-orbit, eclipse, or re-entry)
d. The environmental specifications or life-cycle environmental profiles for each of the
environmental test zones
e. The overall test philosophy, testing strategy, and test objective for each item, includ-
ing any special tailoring or interpretation of design and testing requirements
f. Required special test equipment, facilities, interfaces, and downtime requirements
2 9

| Test Parameters | Test Tolerance |
| --- | --- |
| 50 to 2000 Hz 2500 to 10000 Hz Overall SPL Note: The statistical degrees of freedom shall be at least 100. | ± 3.0 dB ± 5.0 dB ± 1.5 dB |
| Shock Response Spectrum (Peak Absolute Acceleration, Q = 10) Natural Frequencies Spaced at 1/6-Octave Intervals At or below 3000 Hz Above 3000 Hz Note: At least 50% of the spectrum values shall be greater than the nominal test specification. | ± 6.0 dB + 9.0/–6.0 dB |
| Electromagnetic Compatibility | ± 2 dB |

g. Required test tools, test beds, and specialty items necessary to support testing
h. Standards to be used for the recording of test data on computer-compatible electronic
media, such as disks or magnetic tape, to facilitate automated accumulation and
sorting of data
i. Procedures to guard against damage to test article during transportation handling and
testing
j. The collection of parameters and development of a database during testing of units,
subsystems, and at the vehicle level to be used for trending and test effectiveness
assessment.
4.8.2 Test Procedures
Tests shall be conducted using documented test procedures prepared for performing all of the
required tests in accordance with the test objectives in the approved test plans. The test objectives,
testing criteria, and pass/fail criteria shall be stated clearly in the test procedures. The test procedures
shall cover all operations in enough detail so that there is no doubt as to the execution of any step.
Test objectives and criteria shall be stated to relate to design or operations specifications. Minimum/
maximum requirements for valid data and pass/fail criteria shall be verified in the test procedure.
Traceability shall be provided from the specifications or requirements to the test procedures. Where
practical, the individual procedure step that satisfies the requirement shall be identified. The test
procedure for each item shall include, as a minimum, descriptions of the following:
a. Criteria, objectives, assumptions, and constraints
b. Test setup
c. Initialization requirements
d. Input data
e. Test instrumentation
f. Expected intermediate test results
g. Requirements for recording output data
h. Expected output data
i. Minimum/maximum requirements for valid data to consider the test successful
j. Pass/fail criteria for evaluating results, including uncertainty constraints
k. Safety considerations and hazardous conditions
4.9 Documentation
4.9.1 Test Documentation Files
The test plans and procedures including a list of test equipment, calibration dates and uncertainty,
computer software, test data, test log, test results and conclusions, test discrepancies or deficiencies,
3 0

operating time/cycles, pertinent analyses, and resolutions shall be documented and maintained. The
applicable contractors shall maintain the test documentation file for the duration of their contracts.
4.9.2 General Test Data
Pertinent test data shall be maintained in a quantitative form to permit the evaluation of performance
under the various specified test conditions.
4.9.3 Qualification, Protoqualification, and Acceptance
For qualification, protoqualification, and acceptance tests, a summary of the test results shall be doc-
umented in test reports. The test report shall state the degree of success in meeting the test objectives
and shall document and summarize the test results, deficiencies, problems encountered, and problem
resolutions. The responsible contractor design engineer shall certify the accuracy of the results.
4.9.4 Test Log
Formal test conduct shall be documented in a test log. The test log shall identify the personnel
involved and be time recorded to permit a reconstruction of test events such as start time, stop time,
anomalies, and any periods of interruption. (See Reference 1.)
4.9.5 Test Discrepancy
Anomalies, discrepancies, and failures occurring during test activities shall be documented and dispo-
sitioned as specified in the contractor’s approved quality control plan. (See Reference 13.)
4.10 Exceptions to General Requirements
4.10.1 Qualification and Protoqualification by Similarity (QBS)
A unit may be a candidate for qualification, or protoqualification, by similarity to a heritage unit that
has already been qualified, or protoqualified, for launch vehicle, upper stage or space vehicle use. The
following conditions shall be met in order to apply QBS:
a. Heritage unit was not qualified by similarity or analysis.
b. Heritage unit was a representative flight article.
c. The environments, both amplitude and duration, encountered by the heritage unit
during its qualification or flight history are equal to, or more severe than, the
qualification environments intended for the candidate QBS unit.
d. The candidate QBS unit and the heritage units are produced by the same manu-
facturer in the same facility using identical tools, manufacturing processes, qual-
ity control procedures, and manufacturing staff training/certification levels, with-
out gaps that impact in production continuity.
3 1

e. Heritage unit shall have successfully passed a post-environmental functional test
series, without need for waivers, associated with performance indicating survival
of the qualification stresses.
f. The candidate QBS unit and the heritage units shall perform similar functions,
and the heritage unit shall have equivalent or greater operating life with varia-
tions only in terms of performance such as accuracy, sensitivity, formatting, and
input-output characteristics.
g. Supporting documentation for the heritage unit is available and includes spec-
ifications, drawings, qualification test procedures, qualification and acceptance
test reports, ground and flight discrepancy reports with closure history, test waiv-
ers, and flight history summary.
h. Modified units may be a minor variation of the heritage unit. Dissimilarities of
safety, reliability, maintainability, weight, thermal effects, dynamic response, and
structural, mechanical and electrical configurations shall require that the candi-
date QBS unit characteristics be enveloped by the characteristics of the heritage
unit. Minor design changes involving substitution of piece parts and materials
with equivalent reliability items can generally be accepted. Design dissimilarities
resulting from addition or subtraction of piece parts and particularly moving
parts, ceramic or glass parts, crystals, magnetic devices, and power conversion or
distribution equipment shall void qualification by similarity, unless the contrac-
tor’s QBS analysis includes technical rationale supporting the similarity claim.
i. In some cases, the item to be qualified by similarity is not a unit but is some other
level of assembly, such as a subsystem. In that case, the criteria for the item to be
qualified by similarity would be the same as though the item were a unit.
Deviations from these conditions shall require customer approval of the assessment and
documentation.
4.10.2 Electrical and Electronic Unit Thermal Vacuum Test Exemptions
Requirements for unit testing are provided in Tables 6.3-1 and 6.3-2. For electronic units, there may
be instances where unit-level thermal vacuum testing is unnecessary if it can be shown that the design
is insensitive to the vacuum environment. Deletion of the thermal vacuum test when this or other tai-
lored criteria demonstrate unit vacuum insensitivity is restricted to electronic units only (not mechani-
cal units) and for acceptance testing only. A criterion delineating the conditions under which vacuum
testing may be exempted needs to be defined early in the program to allow for test planning and risk
mitigation activities. As a minimum, the criteria used in assessing vacuum-sensitivity in electronic
units should include consideration for confidence gained from identical flight units, the vacuum-
sensitive nature of any high-voltage or radio frequency (RF) units, susceptibility for arcing, thermal
control features that need to be verified in vacuum, deflection of any sealed devices, and electrical or
thermal performance issues that may be different under vacuum conditions, such as:
3 2

a. Units that have no flight heritage or do not have a qualification unit thermal vacuum
test associated with their design shall be tested in a vacuum environment.
b. Units that are inherently vacuum and/or temperature sensitive, such as RF equipment,
shall be tested in a vacuum environment.
c. Units susceptible to corona or multipaction, such as high-voltage units, shall be tested
in a vacuum environment.
d. Devices that are temperature controlled on-orbit to within a range of 3°C or less to
maintain performance shall be tested in a vacuum environment.
e. If a hermetically sealed device can physically deflect under worst-case conditions
such that the clearance between the device wall and a nearby item could cause an
electrical short (e.g., a clearance of 2.5 mm or less), the unit shall be tested in a vac-
uum environment.
f. Units whose electrical performance may be affected by temperature differences
between the ambient and vacuum thermal environments shall be tested in a vacuum
environment. If the unit performance in vacuum and ambient conditions has been
well characterized by test data such that performance problems can be clearly
detected in ambient testing, deletion of the vacuum environment may be considered.
g. A unit with any part case temperature prediction within 10ºC of its allowable derated
temperature limit under worst-case power dissipation shall be tested in a vacuum
environment.
h. Unit thermal analyses shall be performed in vacuum and ambient environments to
assess vacuum-sensitivity.
1. If the worst-case temperature difference between the two environments is greater
than 10oC, thermal vacuum testing shall be performed.
2. If the worst-case temperature difference is between 3ºC and 10ºC and all criteria
support ambient testing in lieu of vacuum testing, the base plate temperature shall
be increased during the ambient test to account for junction/case temperature dif-
ferences between the two environments.
3. If the worst-case temperature difference is less than 3ºC and all criteria support
ambient testing in lieu of vacuum testing, thermal cycling may be performed in
lieu of thermal vacuum testing without base plate temperature modification.
If a unit’s qualification test was performed with a baseplate temperature higher than 10ºC
above the acceptance baseplate test temperature, then the 10ºC limit specified above may
be increased to the corresponding higher baseplate temperature margin.
3 3

THIS PAGE INTENTIONALLY LEFT BLANK

5. Alternative Strategies
The qualification testing in 4.3.2 provides a demonstration that the design, manufacturing, and
acceptance testing produces flight items that meet specification requirements. In a minimum-risk
program, the hardware items subjected to qualification tests are not eligible for flight without careful
analysis and refurbishment since the remaining life from fatigue and wear standpoints is not demon-
strated. In programs where it is necessary to demonstrate the flightworthiness of tested hardware, the
strategies described in 5.1 and 5.2 as alternatives to the protoqualification strategy may be used at the
system, subsystem, and unit levels. A combination of various applications of qualification and pro-
toqualification strategies should be considered to meet the needs for particular items, as deemed nec-
essary. As in the case for protoqualification, the higher risk of deviating from qualification may be
partially mitigated by enhanced development testing and by increasing the design margins. Care
should be exercised to ensure that an acceptable balance of demonstrated margin, workmanship
screening, and remaining life for flight is achieved for programs pursuing these options.
5.1 Spares Strategy
This strategy does not alter the qualification and acceptance test requirements presented in 4.3.2 and
4.3.4. Yet, in some cases, qualification hardware may be used for flight if the risk is minimized.
Typically, the qualification test program results in a qualification test vehicle that was built using
units that had been qualification tested at the unit level. After completing the qualification tests, the
critical units can be removed from the vehicle, and the qualification vehicle can then be refurbished,
as necessary. Usually a new set of critical units would be installed that had only been acceptance
tested. This refurbished qualification vehicle would then be certified for flight when it satisfactorily
completed the vehicle acceptance tests. The qualification units that were removed could be refur-
bished and used as flight spares.
5.2 Flightproof Strategy
This strategy applies only for dynamic environments. It subjects all flight items to a flightproof test
that is an enhanced acceptance test using protoqualification levels, while retaining acceptance dura-
tion. The risk taken is that there has been no formal demonstration of remaining life for flight and
limited demonstration of capability to withstand the flight environment. This strategy does not vali-
date subsequent acceptance testing. Flightproof testing is applicable to single vehicle procurements.
Flightproof testing is a check on the adequacy of the capability of each flight item, considering build
variability or defects introduced due to handling or testing. Development testing, design to enhanced
margins, and rigorous analysis should be used to gain confidence that adequate margin remains after
the maximum allowed accumulation of preflight testing. For thermal testing, flightproof hardware
shall be tested to protoqualification requirements.
3 5

5.3 Combination Test Strategies
Various combinations of test strategies may be considered, depending on specific program considera-
tions and the degree of risk deemed acceptable. For example, the protoqualification strategy for units
(4.3.3) may be combined with the flightproof strategy for the vehicle. In other cases, the flightproof
strategy would be applied to some units peculiar to a single mission, while the protoqualification
strategy may be applied to multi-mission units. In such cases, the provisions of each method would
apply, and the resultant risk would be increased correspondingly.
5.4 Qualification Using Engineering Qualification Models (EQM)
The use of EQM hardware to achieve qualification testing may be acceptable in cases where hardware
configuration and successful test exposure provide demonstration of desired qualification of the
hardware. In those cases where the EQM hardware differs from flight hardware, a quantitative risk
assessment shall be performed and documented. See Reference 15 for additional guidance.
The EQM shall have, as a minimum, the following characteristics:
a. Use of flight-like parts (same part numbers as flight units; up-screening not required)
b. Use of flight-like boards, slices and housings
c. All redundancies included
d. All boards and slices must be present in the EQM
e. All qualification testing shall occur on the same EQM
f. EQM built using flight build processes, personnel, and skill set in the same factory as flight
units. Production gaps between the EQM qualification and acceptance manufacturing shall
be assessed.
g. EQM tested with same verification and validation approach, and test equipment as flight units.
Deviations from these conditions shall require assessment and documentation.
The EQM shall meet all flight hardware requirements after exposure to qualification environments.
An assessment of the differences between the EQM and the flight units requires customer approval
before commencement of flight hardware manufacturing.
3 6

6. Unit Test Requirements
6.1 General Requirements
Unit tests shall normally be accomplished entirely at the unit level. However, in certain circum-
stances where one or more units are needed to complete a function, the required unit tests may be
conducted at the next level of assembly. Tests of units such as interconnect tubing, radio-frequency
circuits, and wiring harnesses are examples where at least some of the tests may be accomplished at
higher levels of assembly. If moving mechanical assemblies or other units have static or dynamic
fluid interfaces or are pressurized during operation, those conditions should be replicated during unit
testing. Units shall meet the applicable requirements over the entire qualification and acceptance
environmental test range.
6.2 Development Tests
6.2.1 Subassembly Development Tests, In-Process Tests, and Inspections
Subassemblies are subjected to development tests and evaluations as required to minimize design
risk, to demonstrate manufacturing feasibility, and to assess the design and manufacturing alternatives
and trade-offs required to best achieve the development objectives. Tests are conducted as required
to develop in-process manufacturing tests, inspections, and acceptance criteria for the items. Oppor-
tunity to establish subassembly qualification should be exploited when appropriate. For example, it is
often easier to demonstrate design margin and performance at lower levels of assembly, especially in
cases where existing designs are integrated into a new and unique next assembly, which will then be
re-qualified.
6.2.2 Unit Development Tests
Units are subjected to development tests and evaluations as may be required to minimize design risk,
to demonstrate manufacturing feasibility, to establish packaging designs, to demonstrate electrical and
mechanical performance, and to demonstrate the capability to withstand environmental stress,
including storage, transportation, extreme combined environments, and launch base operations.
6.2.3 Development Tests for Composite Structures
Development tests should be conducted on structural components made of composite and/or bonded
materials used in space vehicles, interstages, payload adapters, payload fairings, motor cases, nozzles,
propellant tanks, and over-wrapped pressure vessels.
If appropriate, testing should include:
a. Demonstration of manufacturing feasibility
b. Static load or burst testing to verify analysis models and design margin predictions
c. Damage tolerance testing to define acceptance criteria for fracture critical structural items
3 7

d. Demonstration of scalability of basic design data to full scale design
e. Characterization of mechanical properties (strength, stiffness, acoustic loss, etc.) and
electrical conductivity
f. Characterization of venting and other relevant material properties, such as material
porosity, density, and outgassing. For perforated sandwich structures, if venting at the
flight depressurization rate cannot be demonstrated, tests should be conducted to verify the
unit has sufficient strength to withstand internal pressure.
6.2.4 Thermal Development Tests
For critical electrical and electronic units designed to operate in a vacuum environment less than
0.133 Pa (0.001 Torr), thermal mapping for known boundary conditions may be performed in the
vacuum environment to verify the internal unit thermal analysis, and to provide data for thermal
mathematical model correlation. Once correlated, the thermal model is used to demonstrate that criti-
cal part temperature limits, consistent with reliability requirements and performance, are not
exceeded.
When electrical and electronic packaging is not accomplished in accordance with known and
accepted techniques relative to the interconnect subsystem, parts mounting, board sizes and thickness,
number of layers, thermal coefficients of expansion, or installation method, development tests should
be performed. The tests should establish confidence in the design and manufacturing processes.
Heat transport capacity tests may be required for constant and variable conductance heat pipes at the
unit level to demonstrate compliance. Thermal conductance tests should be considered to verify con-
ductivity across items such as vibration isolators, thermal isolators, cabling, and any other potentially
significant heat conduction path.
Engineering Development Units (EDU) of cryogenic systems may be tested in a vacuum environment
for early verification of system performance and margins, and assessment of parasitic heat leaks. A
thermal balance test may also be conducted to demonstrate thermal control hardware and subsystem
performance and to collect data for thermal model correlation.
6.2.5 Shock and Vibration Isolator Development Tests
When a unit is to be mounted on shock or vibration isolators whose performance is not well known,
development testing should be conducted to verify their suitability. The isolators should be exposed
to the various induced environments (e.g., temperature and chemical environments) to verify retention
of isolator performance (especially resonant frequencies and amplifications) and to verify that the
isolators have adequate service life. The unit or a rigid simulator with proper mass properties (mass,
center of gravity, mass moments of inertia) should be tested on its isolators in each of three orthogo-
nal axes, and, if necessary, in each of three rotational axes. Responses at all corners of the unit should
be determined to evaluate isolator effectiveness and, when applicable, to establish the criteria for unit
acceptance testing without isolators. When multiple units are supported by a vibration-isolated panel,
responses at all units should be measured to account for the contribution of panel vibration modes.
3 8

6.2.6 Battery Development Tests
See Reference 20 for development test guidance. For power systems using lithium-ion batteries, see
Reference 26 for recommendations on spacecraft batteries and Reference 27 for recommendations on
launch vehicle batteries. Safety testing is required for lithium-ion batteries during qualification and
evaluation of safety devices is required during acceptance testing.
6.2.7 EMI/EMC Development Tests
See Reference 21 for test guidance.
6.3 Test Program for Units
Tables 6.3-1 and 6.3-2 identify unit qualification, protoqualification, and acceptance test require-
ments. When units fall into two or more categories, the most stressing combination of tests shall be
performed. For example, a star sensor may be considered to fit both “Electrical and Electronic” and
“Optical” categories. A thruster with integrated valves would be considered to fit both “Thruster” and
“Valve” categories. Table 6.3-3 provides a summary of unit test level margins and durations that are
discussed in this section.
In all tables shown in this standard where “Evaluation Required” (ER) is noted, an engineering
assessment shall be performed and documented to develop rationale for performing or not performing
a test. The customer will approve the contractor’s evaluation.
Additional requirements are specified in Reference 9 for solar cells and Reference 10 for solar panels.
See 6.2.6 for additional test requirements for batteries.
6.3.1 Unit Wear-In Test
6.3.1.1 Purpose
The wear-in test detects material and workmanship defects that occur early in the unit life. Testing
also serves to wear-in or run-in mechanical units so they perform in a smooth, consistent, and con-
trolled manner.
6.3.1.2 Test Description
While the unit is operating under conditions representative of operational loads, speed, and environ-
ments, and while perceptive parameters are being monitored, the unit shall be operated for the speci-
fied duration or number of cycles.
Pressure components with no moving mechanical assembly aspects, such as filters, are not subject to
wear-in testing. Thruster wear-in is limited to valves, which may be tested prior to integration within
a thruster.
3 9

Table 6.3-1. Unit Qualification and Protoqualification Test Summary
Electrical
Reference Suggested and Solar Battery Pressure Pressure Structural
Test Paragraph Sequence Electronic Antenna MMA Panel (12) Component Vessel Thruster Thermal Optical Components
Inspection(1) 4.6 1, 18 R R R R R R R R R R R
Performance(1) 6.3.2 2, 17 R R R R R R R R R R ER
Leakage 6.3.3 3, 7, 12 ER -- R(11) -- R R R R R -- --
Shock 6.3.4 4 R ER ER ER R(6) ER ER ER ER ER ER
Vibration or 6.3.5 5 R R R R R R R R R R ER
Acoustic(2) 6.3.6
Acceleration 6.3.7 6 ER ER ER ER ER -- ER -- -- ER ER
Thermal Cycle 6.3.8 8 R(7) ER ER ER R ER ER ER ER ER ER(3)
Thermal Vacuum 6.3.9 9 R(7) R R R(9) R ER ER R(8) R R –
Climatic 6.3.10 10 ER ER ER ER ER ER ER ER ER ER ER
Pressure 6.3.12 11 ER -- ER -- R R R ER ER(5) -- --
EMC(4) 6.3.13 13 R ER ER ER ER ER ER ER ER ER ER
Life 6.3.14 14 ER ER R ER R R ER R ER ER ER
Static Load 6.3.11 15 ER ER ER ER ER -- ER -- -- -- R
Burst Pressure 6.3.12 16 -- -- ER -- R R(10) R(10) R ER -- --
R Required (6) Required for launch vehicle and upper stage. Evaluation required for
ER Evaluation required (see 6.3) space vehicle.
(1) Performance tests shall be conducted prior to, during, and following each (7) Burn-in required for protoqualification units.
environmental test, as appropriate. (8) Thruster thermal vacuum requirement may be satisfied with hot fire
(2) Either vibration or acoustic required, with the other discretionary. testing.
(3) Required on composite and/or bonded structural components. (9) Thermal vacuum testing of solar panels may be performed on solar
(4) Required for non-electrical/electronic units when they provide required arrays.
electromagnetic shielding, when there is a passive intermodulation (10) Burst pressure testing is required for qualification testing only.
requirement or when the unit contains active or passive electrical (11) Required only for pressurized and hermetically sealed MMAs.
components. (12) Safety device evaluation required for lithium-ion batteries.
(5) Required for sealed and pressurized units (such as heat pipes).
Evaluation required for other components.
4 0
40

| Test | Reference Paragraph | Suggested Sequence | Electrical and Electronic | Antenna | MMA | Solar Panel | Battery (12) | Pressure Component | Pressure Vessel | Thruster | Thermal | Optical | Structural Components |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Inspection(1) | 4.6 | 1, 18 | R | R | R | R | R | R | R | R | R | R | R |
| Performance(1) | 6.3.2 | 2, 17 | R | R | R | R | R | R | R | R | R | R | ER |
| Leakage | 6.3.3 | 3, 7, 12 | ER | -- | R(11) | -- | R | R | R | R | R | -- | -- |
| Shock | 6.3.4 | 4 | R | ER | ER | ER | R(6) | ER | ER | ER | ER | ER | ER |
| Vibration or Acoustic(2) | 6.3.5 6.3.6 | 5 | R | R | R | R | R | R | R | R | R | R | ER |
| Acceleration | 6.3.7 | 6 | ER | ER | ER | ER | ER | -- | ER | -- | -- | ER | ER |
| Thermal Cycle | 6.3.8 | 8 | R(7) | ER | ER | ER | R | ER | ER | ER | ER | ER | ER(3) |
| Thermal Vacuum | 6.3.9 | 9 | R(7) | R | R | R(9) | R | ER | ER | R(8) | R | R | – |
| Climatic | 6.3.10 | 10 | ER | ER | ER | ER | ER | ER | ER | ER | ER | ER | ER |
| Pressure | 6.3.12 | 11 | ER | -- | ER | -- | R | R | R | ER | ER(5) | -- | -- |
| EMC(4) | 6.3.13 | 13 | R | ER | ER | ER | ER | ER | ER | ER | ER | ER | ER |
| Life | 6.3.14 | 14 | ER | ER | R | ER | R | R | ER | R | ER | ER | ER |
| Static Load | 6.3.11 | 15 | ER | ER | ER | ER | ER | -- | ER | -- | -- | -- | R |
| Burst Pressure | 6.3.12 | 16 | -- | -- | ER | -- | R | R(10) | R(10) | R | ER | -- | -- |

Table 6.3-2. Unit Acceptance Test Summary
Electrical
Reference Suggested & Solar Battery Pressure Pressure Structural
Test Paragraph Sequence Electronic Antenna MMA Panel (12) Component Vessel Thruster Thermal Optical Components
Inspection(1) 4.6 1, 15 R R R R R R R R R R R
Wear-in 6.3.1 2 -- -- R -- ER R -- R -- -- --
Performance(1) 6.3.2 3, 14 R R R R R R R R R R ER
Leakage 6.3.3 4, 7, 12 ER -- R(11) -- R R R R ER (4) -- --
Shock 6.3.4 5 ER -- ER -- ER ER -- ER -- ER --
Vibration 6.3.5
or 6 R R R R R(6) R ER R ER(3) R ER
Acoustic(2)
6.3.6
Thermal Cycle 6.3.8 8 R(8) ER ER ER ER ER ER ER ER ER ER(3)
T V h a e c r u m um al 6.3.9 9 R(7)(8) R R R(10) R(6) ER ER R(9) R R --
Proof 6.3.12 10 ER -- ER -- R R R ER ER(4) -- --
Pressure
Proof Load 6.3.11 11 -- ER ER -- ER -- ER -- -- ER R(3)
EMC (5) 6.3.13 13 ER ER -- ER ER ER -- -- -- -- --
R Required (7) See 4.10.2 for exemptions.
ER Evaluation required (see 6.3). (8) Burn-in required.
(1) Performance tests shall be conducted prior to, during, and after each (9) Thruster thermal vacuum requirement may be satisfied with hot fire testing.
environmental test, as appropriate. (10) Thermal vacuum testing of solar panels may be performed on solar arrays.
(2) Vibration or acoustic required, with the other discretionary. (11) Required only for pressurized or hermetically sealed units.
(3) Required only on composite and/or bonded structures. (12) Safety device testing required for lithium-ion batteries.
(4) Required for sealed or pressurized units (such as heat pipes).
(5) Required when there is less than 12 dB margin.
(6) Evaluation required for silver-zinc batteries.
4 1
41

| Test | Reference Paragraph | Suggested Sequence | Electrical & Electronic | Antenna | MMA | Solar Panel | Battery (12) | Pressure Component | Pressure Vessel | Thruster | Thermal | Optical | Structural Components |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Inspection(1) | 4.6 | 1, 15 | R | R | R | R | R | R | R | R | R | R | R |
| Wear-in | 6.3.1 | 2 | -- | -- | R | -- | ER | R | -- | R | -- | -- | -- |
| Performance(1) | 6.3.2 | 3, 14 | R | R | R | R | R | R | R | R | R | R | ER |
| Leakage | 6.3.3 | 4, 7, 12 | ER | -- | R(11) | -- | R | R | R | R | ER (4) | -- | -- |
| Shock | 6.3.4 | 5 | ER | -- | ER | -- | ER | ER | -- | ER | -- | ER | -- |
| Vibration or Acoustic(2) | 6.3.5 6.3.6 | 6 | R | R | R | R | R(6) | R | ER | R | ER(3) | R | ER |
| Thermal Cycle | 6.3.8 | 8 | R(8) | ER | ER | ER | ER | ER | ER | ER | ER | ER | ER(3) |
| Thermal Vacuum | 6.3.9 | 9 | R(7)(8) | R | R | R(10) | R(6) | ER | ER | R(9) | R | R | -- |
| Proof Pressure | 6.3.12 | 10 | ER | -- | ER | -- | R | R | R | ER | ER(4) | -- | -- |
| Proof Load | 6.3.11 | 11 | -- | ER | ER | -- | ER | -- | ER | -- | -- | ER | R(3) |
| EMC (5) | 6.3.13 | 13 | ER | ER | -- | ER | ER | ER | -- | -- | -- | -- | -- |

Table 6.3-3. Unit Test Level Margins and Duration
Test Qualification Protoqualification Acceptance
Shock(1) 6 dB above MPE, 3 times in each of 3 3 dB above MPE, 2 times in each of 3 Maximum predicted environment (MPE),
orthogonal axes orthogonal axes once in each of 3 orthogonal axes
Acoustic(2) 6 dB above acceptance for 3 dB above acceptance for Envelope of MPE and minimum
3 minutes 2 minutes spectrum (Figure 6.3.6-1) for 1 minute
Vibration(2) 6 dB above acceptance for 3 dB above acceptance for Envelope of MPE and minimum
3 minutes in each of 3 axes 2 minutes in each of 3 axes spectrum (Figure 6.3.5-1) for 1 minute in
each of 3 axes
Thermal Vacuum (non-electrical and ±10°C beyond acceptance for ±5°C beyond acceptance for MPT for 1 cycle
non-electronic) 6 cycles 3 cycles
Thermal Cycle ±10°C beyond acceptance for ±5°C beyond acceptance for Envelope of MPT and minimum range
or Thermal Vacuum Only 27 cycles 20 cycles (–24 to 61°C) for 14 cycles
Combined Thermal Vacuum and ±10°C beyond acceptance for ±5°C beyond acceptance for Envelope of MPT and minimum range
Thermal Cycle 4 thermal vacuum cycles and 4 thermal vacuum cycles and (–24 to 61°C) for 4 thermal vacuum
23 thermal cycles 16 thermal cycles cycles with minimum 2-hour hot
operational dwell and 10 thermal cycles
Static Load(3) 1.25 times the limit load for unmanned 1.25 times the limit load for unmanned 1.1 times the limit load
flight or 1.4 times limit load for manned flight or 1.25 times limit load for manned
flight flight,
Pressure(4) Pressures as specified in Not applicable 1.1 times MEOP for pressurized
Table 6.3.12-2 following acceptance structures; 1.25 times MEOP for
proof pressure test, duration sufficient to pressure vessels; 1.5 times MEOP for
collect data pressure components. Other metallic
pressurized hardware items per
References 4 and 5
EMC 12 dB minimum duration same as 6 dB minimum duration same as 6 dB 20 minutes at each space vehicle
acceptance acceptance transmitter frequency for radiated
susceptibility
(1) See B.1.6 for additional information.
(2) See B.1.2 and B.1.3 for units with effective duration greater than 15 seconds.
(3) See References 6 and 11.
(4) See Reference 19 for solid rocket motor cases used in expendable launch vehicles.
4 2
42

| Test | Qualification | Protoqualification | Acceptance |
| --- | --- | --- | --- |
| Shock(1) | 6 dB above MPE, 3 times in each of 3 orthogonal axes | 3 dB above MPE, 2 times in each of 3 orthogonal axes | Maximum predicted environment (MPE), once in each of 3 orthogonal axes |
| Acoustic(2) | 6 dB above acceptance for 3 minutes | 3 dB above acceptance for 2 minutes | Envelope of MPE and minimum spectrum (Figure 6.3.6-1) for 1 minute |
| Vibration(2) | 6 dB above acceptance for 3 minutes in each of 3 axes | 3 dB above acceptance for 2 minutes in each of 3 axes | Envelope of MPE and minimum spectrum (Figure 6.3.5-1) for 1 minute in each of 3 axes |
| Thermal Vacuum (non-electrical and non-electronic) | ±10°C beyond acceptance for 6 cycles | ±5°C beyond acceptance for 3 cycles | MPT for 1 cycle |
| Thermal Cycle or Thermal Vacuum Only | ±10°C beyond acceptance for 27 cycles | ±5°C beyond acceptance for 20 cycles | Envelope of MPT and minimum range (–24 to 61°C) for 14 cycles |
| Combined Thermal Vacuum and Thermal Cycle | ±10°C beyond acceptance for 4 thermal vacuum cycles and 23 thermal cycles | ±5°C beyond acceptance for 4 thermal vacuum cycles and 16 thermal cycles | Envelope of MPT and minimum range (–24 to 61°C) for 4 thermal vacuum cycles with minimum 2-hour hot operational dwell and 10 thermal cycles |
| Static Load(3) | 1.25 times the limit load for unmanned flight or 1.4 times limit load for manned flight | 1.25 times the limit load for unmanned flight or 1.25 times limit load for manned flight, | 1.1 times the limit load |
| Pressure(4) | Pressures as specified in Table 6.3.12-2 following acceptance proof pressure test, duration sufficient to collect data | Not applicable | 1.1 times MEOP for pressurized structures; 1.25 times MEOP for pressure vessels; 1.5 times MEOP for pressure components. Other metallic pressurized hardware items per References 4 and 5 |
| EMC | 12 dB minimum duration same as acceptance | 6 dB minimum duration same as acceptance | 6 dB 20 minutes at each space vehicle transmitter frequency for radiated susceptibility |

6.3.1.3 Test Levels and Duration
a. Pressure. Ambient pressure is normally used (see 3.2).
b. Temperature. Ambient temperature shall be used for operations if the test objectives
can be met. Otherwise, temperatures representative of the operational environment shall
be used.
c. Duration. The run-in, or wear-in, test shall be performed as indicated in Reference 7.
It consists of at least five cycles or 5% of the total expected service life (3.44), whichever
is greater, unless the unit demonstrates the capability to perform in a predictably
consistent and controlled manner with fewer cycles.
Additional requirements for MMAs are provided in Reference 7.
6.3.1.4 Supplementary Requirements
Perceptive parameters shall be monitored during the wear-in test to detect evidence of functional or
performance degradation.
6.3.2 Unit Performance and Functional Test
6.3.2.1 Purpose
The performance test verifies that the unit meets the requirements of the unit specification. Func-
tional testing verifies that the unit is operational. For MMAs see Reference 7 for functional test
description.
6.3.2.2 Electrical Test Description
Electrical tests shall include application of expected voltages, impedances, frequencies, pulses, and
waveforms (commands, data, clocking, polarity, etc.) at the electrical interfaces of the unit, including
all redundant circuits. These parameters shall be varied throughout their specification ranges and the
sequences expected in flight operation. The unit output and response shall be measured to verify that
the unit performs to requirements. Performance shall also include electrical continuity, response
time, or other tests that relate to a particular unit design. Harness specific testing is required by Ref-
erence 25.
6.3.2.3 Mechanical Test Description
Testing of moving mechanical assemblies shall be performed according to Reference 7.
For other unit types, tests shall be conducted to demonstrate that the unit is capable of operating such
that all performance requirements are satisfied. These tests shall be conducted before, during, and
after exposure to environmental test conditions and shall include measurements to determine whether
performance specifications are met and whether damage or degradation in performance has occurred.
4 3

6.3.3 Unit Leakage Test
6.3.3.1 Purpose
The leakage test demonstrates the capability of pressurized components and hermetically sealed units
to meet the specified design leakage rate requirements.
6.3.3.2 Test Description
An acceptable leak rate to meet mission requirements is based upon development tests and appropri-
ate analyses. An acceptable measurement technique is one that accounts for leak rate variations with
differential pressure and hot and cold temperatures and has the required threshold, resolution, and
accuracy to detect any leakage equal to, or greater than, the maximum acceptable leak rate. Consid-
eration should be given to testing units at differential pressures greater or less than the maximum or
minimum operating differential pressure to provide some assurance of a qualification margin for
leakage. Consideration should also be given to testing at operational temperature extremes using an
appropriate fluid to account for geometry and viscosity changes, as well as leakage under dynamic
conditions during mission operations.
6.3.3.3 Test Level and Duration
The leakage tests shall be performed with the unit pressurized at the maximum differential operating
pressure, as well as at the minimum differential operating pressure if the seals are dependent upon
pressure for proper sealing. If appropriate to sealing capability, the leak rate test shall be made at
qualification hot and cold temperatures with the representative fluid to account for geometry altera-
tions and viscosity changes. If appropriate to mission operations, the leak rate test shall be conducted
under dynamic conditions. The test duration shall be sufficient to detect any significant leakage. This
test shall be performed for qualification, protoqualification, and acceptance testing.
6.3.4 Unit Shock Test
6.3.4.1 Purpose
The qualification or protoqualification shock test demonstrates the capability of the unit to survive
shock and meet requirements during and after exposure to a margin over the maximum predicted
shock environment (MPE) as defined in 3.28. The acceptance shock test demonstrates the workman-
ship of the unit during and after exposure to the maximum predicted shock environment.
6.3.4.2 Test Description
The unit shall be mounted to a fixture through the normal mounting points. The same test fixture
should be used in the qualification, protoqualification, flightproof, and acceptance shock tests. If
shock isolators are to be used in service, they shall be installed. The selected test method shall be
capable of meeting the required shock spectrum with a transient that has duration comparable to the
duration of the expected shock in flight.
A mounting of the unit on an actual, or dynamically similar, flight structure provides a more realistic
test than does a mounting on a rigid structure such as a shaker armature or slip table. Sufficient prior
development of the test mechanism shall have been carried out to validate the proposed test method
4 4

before testing qualification, protoqualification, flightproof, or acceptance flight hardware. The test
environment shall comply with the following conditions:
a. The prescribed shock spectrum can be generated within specified tolerances.
b. The applied shock transient shall approximate the duration and frequency content of the
service shock event.
6.3.4.3 Test Level and Exposure
The shock spectrum in each direction along each of the three orthogonal axes shall meet the test
specification for that direction. For vibration- or shock-isolated units, the lower frequency limit of the
response spectrum shall be below 0.5 times the natural frequency of the isolated unit. The minimum
number of shocks shall be imposed to meet the amplitude criteria in both directions of each of the
three orthogonal axes as follows:
Qualification: MPE + 6 dB applied 3 times
Protoqualification: MPE + 3 dB applied 2 times
Acceptance: MPE for one application
If a shock application cannot concurrently satisfy requirements in both directions of any axis, the test
shall be conducted separately in each direction of that axis.
6.3.4.4 Supplementary Requirements
During shock testing wiring harnesses and hydraulic and pneumatic lines, instrumentation, and other
connected items shall be equivalent to, or simulate the flight configuration to the first attachment
point. The intent of this requirement is to simulate the external interconnecting mass, inertia, and
stiffness effects on the unit’s dynamic response unless shown to be insignificant.
Electrical and electronic units, including redundant circuits, shall be energized and monitored, if
feasible and safe. Functional performance shall be monitored during the unit shock test. A perfor-
mance test shall be performed before and after completion of shock testing.
A shock qualification or protoqualification test shall be required for electrical and electronic units and
batteries for which any of the following conditions apply:
a. Shock MPE response spectrum (SRS) envelopes the unit random vibration spectrum con-
verted to its corresponding acceptance SRS, at any frequency below 2,000 Hz or exceeds 50
in/sec modal velocity at any frequency below 10,000 Hz
b. The unit contains shock-sensitive components. Typical examples are crystals, ceramic chips,
relays, etc.
For batteries see References 20, 26, and 27 for additional considerations.
4 5

6.3.5 Unit Vibration Test
6.3.5.1 Purpose
The vibration qualification and protoqualification tests demonstrate the ability of the unit to endure a
limited duration of acceptance testing and then meet requirements during and after exposure to a mar-
gin over the maximum predicted vibration environment (MPE) in flight. The vibration test may be
discretionary for a unit having a large surface causing its vibration response to be due predominantly
to direct acoustic excitation. Both acoustic and vibration tests are required, if necessary, to demon-
strate the capability to withstand vibration excitation transmitted through its attachments.
6.3.5.2 Test Description
The unit shall be mounted to a test fixture through the normal mounting points of the unit. The same
fixture shall be used in the qualification, protoqualification, and acceptance vibration tests. Attached
wiring harnesses and hydraulic and pneumatic lines, instrumentation, and other connected items shall
be equivalent to, or simulate the flight configuration to the first attachment point. The intent of this
requirement is to simulate the external interconnecting mass, inertia, damping, and stiffness effects on
the unit’s dynamic response, unless shown to be insignificant. These items shall be connected using
flight-like connectors. Such a configuration shall also be required when units that employ shock or
vibration isolators are tested on their isolators. The suitability of the fixture and test control shall
have been established (6.3.5.6) prior to the qualification testing. The unit shall be tested in each of
three orthogonal axes.
Units mounted on shock or vibration isolators shall typically require vibration testing at qualification
levels in two configurations. A first configuration is with the unit hard-mounted to qualify for the
acceptance-level testing if, as is typical, the acceptance testing is performed without the isolators pre-
sent. The second configuration is with the unit mounted on the isolators to qualify for the flight envi-
ronment. The unit shall be mounted on isolators of the same production lot as those used in service.
Units mounted on isolators shall be controlled at the locations where the isolators are attached to the
fixture. Hard-mounted units shall be controlled at the unit mounting attachment or attachments as
appropriate.
6.3.5.3 Test Levels and Duration
The basic test levels and duration required for units exposed to the liftoff and ascent vibration, effec-
tive duration of 15 seconds (see 3.12), are as follows:
Qualification: 6 dB above acceptance for 3 min/axis (B.1.1, B.1.2)
Protoqualification: 3 dB above acceptance for 2 min/axis (B.1.1, B.1.2)
Acceptance: Envelope of MPE and minimum level shown in Figure 6.3.5-1 for
1 min/axis; for units heavier than 50 lbs, see B.2 for minimum
level rationale
The qualification test demonstrates that adequate life remains for flight units after up to eight minutes
of acceptance testing for each axis. The protoqualification test demonstrates that adequate life
remains for subsequent flight units after only one minute of acceptance-level testing for each axis.
However, the protoqualification test does not demonstrate adequate life left for flight of the
4 6

Spectrum Values
Frequency (Hz) Minimum PSD (g2/Hz)
20 0.0053
20 to 150 +3 dB per octave slope
150 to 800 0.04
800 to 2000 -6 dB per octave slope
2000 0.00644
For heavier units see B.2
Figure 6.3.5-1. Minimum random vibration spectrum, unit acceptance test.
protoqualification unit itself. The acceptance test demonstrates quality of workmanship and
performance to specification.
For units subject to exposure to flight vibration longer than 15 seconds, or to demonstrate another
bound on the accumulated duration of acceptance testing, see B.1.3 for changes to the qualification
and protoqualification durations. See B.1.4 for an alternate test strategy.
For qualification and protoqualification testing of units flown on isolators, see 6.3.5.2 and B.1.4.
4 7

| Spectrum Values |  |
| --- | --- |
| Frequency (Hz) | Minimum PSD (g2/Hz) |
| 20 20 to 150 150 to 800 800 to 2000 2000 | 0.0053 +3 dB per octave slope 0.04 -6 dB per octave slope 0.00644 |
| For heavier units see B.2 |  |

Low-level testing shall be performed in each axis before and after the specified vibration tests to
detect any structural changes.
6.3.5.4 Performance Test
During the test, all electrical and electronic units shall be electrically energized and functionally
sequenced through various operational modes to the maximum extent practical. This includes all
primary and redundant circuits, and all circuits that do not operate during launch. Several perceptive
parameters, such as voltage, current, relay contact, software Built In Test (BIT), etc., shall be moni-
tored for failures or intermittent performance during the test. Continuous monitoring of the unit,
including the main bus by a power transient monitoring device, shall be provided to detect intermit-
tent failures.
6.3.5.5 Supplementary Requirements
The vibration spectrum may be reduced, or notched, to prevent unrealistic input forces or unit
responses due to differences in boundary conditions between test and flight. Reductions in the spec-
trum may also be appropriate to account for attenuation due to weight of units exceeding 50 lb (23
kg). See B.2 for additional information.
Units required to operate under pressure during ascent shall be pressurized to simulate flight condi-
tions, from structural and leakage standpoints, and monitored for pressure decay. Units designed for
operation during ascent, and whose maximum or minimum expected temperatures fall outside the
normal temperature range, are candidates for combined vibration and temperature testing. When such
testing is employed, units shall be conditioned to be as close to the worst-case flight temperature as is
practical and monitored for temperature performance during vibration exposure.
For silver-zinc launch vehicle batteries, this test must be performed after full wet stand time and com-
pletion of non-operational thermal cycle testing in order to demonstrate compliance despite corrosion
modes stimulated by long stand times or high temperature.
See B.1.5 for a discussion of a damage-based analysis approach of flight vibration data and its use for
defining an MPE.
6.3.5.6 Fixture Evaluation
The vibration fixture shall be verified by test to uniformly impart motion to the unit under test and to
limit the energy transfer, or crosstalk, from the test axis to the other two orthogonal axes. The cross-
talk levels shall not exceed the input levels for the respective axes. The dynamic test configuration,
fixture, and test article shall be evaluated for crosstalk before initial testing. The fixture shall be re-
evaluated for changes in shaker or orientation of the test configuration.
6.3.5.7 Special Considerations for Structural Units
Vibration acceptance tests of structural units are normally not conducted because the process controls,
inspections, and proof testing that are implemented are sufficient to assure performance and quality.
However, to demonstrate structural integrity of structural units sensitive to fatigue modes with low
margins of safety, a vibration qualification test shall be conducted. The test duration shall be four
4 8

times the fatigue equivalent duration during flight and ground testing. When a structural unit is not
subjected to a static strength qualification test, a brief random vibration qualification test shall be
conducted with an exposure to 3 dB above acceptance. The duration shall be that necessary to
achieve a steady-state response, but not less than ten seconds, to demonstrate that strength require-
ments are satisfied.
6.3.5.8 Options for Vibration Testing
Alternate vibration test techniques may be employed in those cases where baseline procedures do not
satisfy the objectives of the program.
6.3.5.8.1 Two-Phase Testing
The two-phase approach to vibration qualification, or protoqualification, testing is an alternate test
strategy that may be used for:
a. Units that are isolated in flight but acceptance tested without isolators, or by
b. Cases where reduction in the conservatism in accelerated testing (see B.1.2) for
acceptance life is appropriate. This is especially relevant for units that are exposed to
environments for extended periods of time, such as units located close to operating
engines.
The two-phase approach consists of a Phase I test for acceptance life performed at the acceptance
spectrum for the duration required to demonstrate life for flight and expected acceptance testing. This
is followed by a Phase II test at the qualification/protoqualification spectrum for a period four times
the effective duration of MPE in flight. Both tests shall be conducted on the same test specimen.
B.1.4 provides additional details on this approach.
6.3.5.8.2 Alternate Vibration Test for Qualification and Protoqualification
As stated in 4.2.2, qualification/protoqualification testing is performed to verify the capability of the
design to withstand the flight maximum predicted environments with a specified margin and validate
the acceptance test program for subsequent flight hardware. The discussion in B.1.1 indicates that
baseline qualification testing, performed for three minutes duration at +6 dB margin, demonstrates
eight acceptance lives in addition to flight for subsequent flight hardware. Protoqualification testing,
performed for two minutes duration at +3 dB margin, demonstrates life for one acceptance test, and
flight, but does not demonstrate life for retesting. In some cases it may be possible to qualify the
hardware at intermediate levels and durations, between baseline qualification and protoqualification
to demonstrate capability to withstand stress levels and durations higher than protoqualification but
less stringent than qualification.
As an example consider qualification testing involving vibration exposure at +4.5 dB for 2.5 minutes.
This demonstrates enhanced stress capability and additional acceptance lives compared to protoquali-
fication but increases risk for the tested flight hardware. The +6 dB analytical design margin must be
retained to provide for possible risk mitigation. This option demonstrates life for three acceptance
tests and flight. Two of those lives may be set aside for retesting following possible hardware repairs.
Section B.1.4 provides an explanation of the trades between test levels and durations which may be
used for tailoring this requirement to satisfy program needs.
4 9

Another example may be to increase the exposure time from two to three minutes at acceptance
+3 dB for protoqualification. This option demonstrates life for two acceptance tests plus flight. One
of those acceptance lives may be used for retest.
If any of these options are adopted, the test tolerances shown in Table 4.7-1 shall be adjusted to avoid
degradation of the test to levels below requirements at the low tolerance side of the spectrum. For the
case of the protoqualification example shown above, the test tolerance will be ±0.5 dB between 20 Hz
and 1000 Hz and ±1.5 dB between 1000 Hz and 2000 Hz or as close to these values as the test facility
can achieve.
6.3.6 Unit Acoustic Test
This test is applicable to units with large surface areas that are sensitive to direct acoustic excitation.
6.3.6.1 Purpose
The acoustic qualification and protoqualification tests demonstrate the ability of a unit to endure a
limited duration of acceptance testing and then meet requirements during and after exposure to a mar-
gin over the acceptance test level, which is an envelope of MPE and the minimum acoustic spectrum
shown in Figure 6.3.6-1. Acoustic testing is required for a unit having large surfaces, causing its
vibration response to be due predominantly to direct acoustic excitation. For such units, the vibration
test is discretionary except as noted in 6.3.5.1.
6.3.6.2 Test Description
The unit in its ascent configuration shall be installed in an acoustic test facility capable of generating
sound fields or fluctuating surface pressures that induce unit vibration environments sufficient for unit
qualification. The unit shall be mounted on a flight-like support structure. Significant fluid and
pressure conditions affecting structural damping shall be replicated. Appropriate dynamic instru-
mentation shall be installed to measure vibration and strain responses. Control microphones shall be
placed at a minimum of four well-separated locations at one-half the distance from the test article to
the nearest chamber wall, but no closer than 20 in. to both the test article surface and the chamber
wall. The average of all control microphones shall be used for spectrum control.
6.3.6.3 Test Levels and Duration
The basic test levels and duration required for units exposed to the liftoff and ascent acoustic excita-
tion, effective duration of 15 seconds (see 3.12), are as follows:
Qualification: 6 dB above acceptance for 3 min (see B.1.1, B.1.2)
Protoqualification: 3 dB above acceptance for 2 min (see B.1.1, B.1.2)
5 0

Spectrum Values
1/3 Octave Band Minimum 1/3 Octave Band Minimum
Center Frequency Sound Pressure Center Frequency Sound Pressure
(Hz) Levels (dB) (Hz) Levels (dB)
31.5 121.0 630.0 122.2
40.0 122.0 800.0 120.9
50.0 123.0 1000.0 119.3
63.0 124.0 1250.0 117.5
80.0 124.9 1600.0 115.5
100.0 125.7 2000.0 113.6
125.0 126.3 2500.0 111.9
160.0 126.7 3150.0 110.1
200.0 126.9 4000.0 108.4
250.0 126.6 5000.0 106.5
315.0 126.0 6300.0 104.6
400.0 124.8 8000.0 102.8
500.0 123.6 10000.0 101.1
Overall Sound Pressure Level (OASPL) = 136.8 dB
Figure 6.3.6-1. Minimum acoustic spectrum, unit and vehicle.
Acceptance: Envelope of acoustic MPE (see 3.26) and minimum level shown
in Figure 6.3.6-1 for 1 min
The qualification test demonstrates that adequate life remains for flight units after up to eight minutes
of acceptance testing. The protoqualification test demonstrates that adequate life remains for subse-
quent flight units after only one minute of acceptance-level testing for each axis. However, the pro-
toqualification test does not demonstrate adequate life left for flight of the protoqualification unit
itself. The acceptance test demonstrates quality of workmanship and performance to specification.
5 1

For a longer exposure to flight acoustic excitation, or to demonstrate another bound on the accumu-
lated duration of acceptance testing, see B.1.3 for changes to the qualification and protoqualification
durations. See B.1.4 for an alternate test strategy.
6.3.6.4 Supplementary Requirements
During the test, electrically active units shall be electrically energized and functionally sequenced
through various operational modes to the maximum extent practical. This includes all primary and
redundant circuits, and all circuits that do not operate during launch. Several perceptive parameters,
such as voltage, current, relay contact, software Built In Test (BIT), etc., shall be monitored for fail-
ures or intermittent performance during the test. Continuous monitoring of the unit, including the
main bus by a power transient monitoring device, shall be provided to detect intermittent failures.
See B.1.5 for discussion of a damage-based approach to the analysis of flight acoustic data for deter-
mining the adequacy of established acceptance qualification or protoqualification testing when new
flight data lead to questioning the adequacy of the MPE spectrum.
6.3.6.5 Options for Acoustic Testing
Alternate acoustic test techniques may be employed in those cases where baseline procedures do not
satisfy the objectives of the program.
6.3.6.5.1 Two-Phase Testing
The two-phase approach to acoustic qualification/protoqualification testing is an alternate test strategy
that may be used for:
a. Units that are isolated in flight but acceptance tested without isolators, or
b. Cases where it is desired to avoid the increase in test level required for accelerated testing.
This is relevant for units that are exposed to environments for extended periods of time.
This approach consists of a Phase I test for acceptance life performed with the acceptance spectrum
for the duration required to demonstrate life for flight and expected testing. This is followed by a
Phase II test for flight with the qualification/protoqualification spectrum for a period four times the
effective duration of MPE. Both tests shall be conducted on the same test specimen. See B.1.4 for
further guidance and examples on how to apply this strategy
6.3.6.5.2 Alternate Acoustic Test for Qualification and Protoqualification
As stated in 4.2.2, qualification/protoqualification testing is performed to verify the capability of the
design to withstand the flight maximum predicted environments with a specified margin and validate
the acceptance test program for subsequent flight hardware. The discussion in B.1.1 indicates that
baseline qualification testing, performed for three minutes duration at +6 dB margin, demonstrates
eight acceptance lives in addition to flight for subsequent flight hardware. Protoqualification testing,
performed for two minutes duration at +3 dB margin, demonstrates life for one acceptance test, and
flight, but does not demonstrate life for retesting. In some cases it may be possible to qualify the
hardware at intermediate levels and durations, between baseline qualification and protoqualification
5 2

to demonstrate capability to withstand stress levels and durations higher than protoqualification but
less stringent than qualification.
As an example, consider qualification testing involving acoustic exposure at +4.5 dB for 2.5 minutes.
This demonstrates enhanced stress capability and additional acceptance lives compared to protoquali-
fication but increases risk for the tested flight hardware. The +6 dB analytical design margin must be
retained to provide for possible risk mitigation. This option demonstrates life for three acceptance
tests and flight. Two of those lives may be set aside for retesting following possible hardware repairs.
B.1.4 provides an explanation of the trades between test levels and durations which may be used for
tailoring this requirement to satisfy program needs.
Another example is to increase the exposure time from two to three minutes at acceptance +3 dB for
protoqualification. This option demonstrates life for two acceptance tests plus flight. One of those
acceptance lives may be used for retest. If any of these options are adopted, the test tolerances shown
in Table 4.7-1 shall be adjusted to avoid degradation of the test to levels below requirements at the
low tolerance side of the spectrum. For the case of the protoqualification example shown above, the
test tolerance will be ±0.5 dB between 20 Hz and 1000 Hz and ±1.5 dB between 1000 Hz and 2000
Hz, or as close to these values as the test facility can achieve.
6.3.7 Unit Acceleration Test
6.3.7.1 Purpose
The acceleration test demonstrates the capability of the unit to withstand or, if appropriate, to operate
in the qualification-level acceleration environment. This test shall be performed for qualification and
protoqualification testing.
6.3.7.2 Test Description
The unit shall be attached, as it is during flight, to a test fixture and subjected to acceleration in
appropriate directions. The specified accelerations apply to the center of gravity of the test item. If a
centrifuge is used, the arm (measured to the geometric center of the test item) shall be at least five
times the dimension of the test item measured along the arm. The acceleration gradient across the test
item should not result in accelerations that fall below the qualification level on any critical member of
the test item. In addition, any over-test condition shall be minimized to prevent unnecessary risk to
the test article. Inertial units such as gyros and platforms may require counter-rotating fixtures on the
centrifuge arm. The unit shall be tested in both directions of three orthogonal axes.
6.3.7.3 Test Levels and Duration
a. Acceleration Level. The test acceleration level shall be at least 1.25 times the maximum pre-
dicted acceleration (see 3.25).
b. Duration. Unless otherwise specified, the test duration shall be at least five minutes for each
direction of test.
6.3.7.4 Supplementary Requirements
If the unit is to be mounted on shock or vibration isolators in the vehicle, the unit shall be mounted on
these isolators during the qualification test.
5 3

6.3.8 Unit Thermal Cycle Test
6.3.8.1 Purpose
The thermal cycle test imposes environmental stress screens in an ambient pressure environment to
detect flaws in design, parts, processes, and workmanship. The thermal cycle qualification test dem-
onstrates robustness of the electrical and electronic unit design, operation over the design temperature
range, and the ability to function during subsequent acceptance testing. The thermal cycle acceptance
test demonstrates workmanship integrity and the ability of the unit to survive and operate properly in
the maximum expected conditions of its life cycle.
6.3.8.2 Test Description for Electrical and Electronic Units
With the unit operating (power on), the unit shall be tested while subjected to the temperature profiles
in Figures 6.3.8-1, 6.3.8-2, and 6.3.8-3, or as appropriate, for first, intermediate, and last cycle testing,
respectively. The test control temperature shall be measured at a representative location on the unit,
such as at the mounting point on the baseplate.
For testing at the hot temperature on the first cycle per Figure 6.3.8-1:
Figure 6.3.8-1. Notional temperature profile for first cycle testing.
 The environment (chamber temperature) and unit power are set to ramp the unit to its hot sur-
vival test temperature. If the hot survival temperature is an operational survival limit, the unit
shall be operational (turned on either at ambient or at a hot turn-on temperature) during this tran-
sition. If the hot survival temperature is a non-operational limit, the unit shall not be operating
during this transition.
5 4

 At the survival temperature, time shall be accrued to allow internal unit locations to reach the
survival temperature (thermal dwell).
 The unit shall be soaked at the survival temperature.
 The temperature shall be transitioned to the hot test temperature (e.g., qualification, protoquali-
fication, or acceptance). If the hot survival temperature is a non-operational limit, the unit shall
be turned on at the hot turn-on temperature during the transition to the hot test temperature.
 When the control temperature is within the test tolerance, the environment shall be adjusted to
bring the control temperature to the hot test temperature.
 Additional time shall be accrued at the hot test temperature to allow internal unit locations to
reach the test temperature (thermal dwell).
 Following this dwell, the unit shall be turned off for at least 30 minutes off to allow internal
temperatures to stabilize to non-operational levels. During the non-operational time, the envi-
ronment may be adjusted to keep the unit temperature within the test tolerance.
 The unit shall then be turned on, and if necessary, the environment shall be adjusted to re-
stabilize the unit at the test temperature.
 Performance testing at the hot test temperature shall be conducted.
 After the hot operational soak time is satisfied and performance testing is completed, the envi-
ronment shall be set to ramp the unit to the cold survival temperature.
For testing at the cold temperature on the first cycle per Figure 6.3.8-1:
 To aid in the transition to the cold temperature, the unit may be powered off at the completion
of hot performance testing. If the cold survival temperature is an operational survival limit, the
unit shall be turned on at the cold turn-on temperature or, if not specified, a temperature that is no
colder than the test tolerance above the cold survival temperature. If the cold survival tempera-
ture is a non-operational limit, the unit shall not be operating during this transition.
 At the cold survival temperature, time shall be accrued to allow internal unit locations to reach
the survival temperature (thermal dwell).
 The unit shall be soaked at the survival temperature.
 The temperature shall be transitioned to the cold test temperature. If the cold survival temper-
ature is a non-operational limit, the unit shall be turned on at the cold turn-on temperature during
the transition to the cold test temperature.
5 5

 When the control temperature is within the test tolerance, the environment shall be adjusted to
bring the control temperature to the cold test temperature.
 Additional time shall be accrued at the hot test temperature to allow internal unit locations to
reach the test temperature (thermal dwell).
 Following this dwell, the unit shall be turned off for at least 30 minutes off to allow internal
temperatures to stabilize to non-operational levels. During the non-operational time, the envi-
ronment may be adjusted to keep the unit temperature within the test tolerance.
 The unit shall then be turned on, and, if necessary, the environment shall be adjusted to re-
stabilize the unit at the test temperature.
 Performance testing at the cold test temperature shall be conducted.
 After cold performance testing is completed, the environment shall be set to ramp the unit to
the hot test temperature on cycle 2.
Temperature change from ambient to hot, to cold, and return to ambient constitutes one thermal cycle.
On the first cycle, the unit shall demonstrate survival capability requirements at its survival hot and
survival cold temperatures (see 3.55). Following a thermal dwell at the survival temperature, the unit
shall be maintained at the survival temperature for a minimum of one hour. For a unit with a hot or
cold turn-on temperature requirement different from its operational value, turn-on capability at the
turn-on temperature shall be demonstrated on the first cycle.
Testing at the hot and cold temperatures on intermediate cycles (Figure 6.3.8-2) is similar to that
described for the first cycle except there is no survival demonstration (steps ,, , , , and ),
there are no hot/cold starts (steps , , , and ) and functional testing replaces performance
testing.
Testing at the hot and cold temperatures on the last cycle (Figure 6.3.8-3) is similar to that described
for the first cycle except there is no survival demonstration (steps , , , ,  and ). At the
completion of cold performance testing, the environment is adjusted to return the unit to ambient.
5 6

Figure 6.3.8-2. Notional temperature profile for intermediate cycle testing.
Figure 6.3.8-3. Notional temperature profile for last cycle testing.
5 7

Compliance to specified performance requirements shall be required over acceptance, protoqualifi-
cation, and qualification temperature ranges. Performance tests shall be conducted after electrical and
electronic unit temperatures have stabilized (see 3.57) at the hot and cold temperatures during the first
and last cycle, and at ambient temperature prior to and following the test. Functional tests shall be
performed at hot and cold temperature plateaus on intermediate cycles. These tests shall include:
a. During performance and functional tests, the unit shall be cycled through a sufficient number
of operational modes to fully characterize the performance and functionality of the unit.
b. Performance tests shall test all paths, including continuity, with functional testing as a subset
of performance testing that focuses on critical, primary, and redundant paths to check
functionality.
c. During the test, perceptive parameters shall be monitored for failures, degradation trends, and
intermittent behavior.
d. All electrical circuits and all paths shall be verified for circuit performance and continuity.
e. For units with internal redundancy, performance testing, functional testing, and hot and cold
starts shall be demonstrated on primary and redundant circuits and paths.
f. During temperature transitions, the unit shall be powered on and monitored, except as noted,
and the health of the unit shall be monitored and key parameters trended. Any performance
of the unit required during temperature transitions shall be tested during the test transitions.
6.3.8.3 Test Levels and Duration for Electrical and Electronic Units
a. Pressure and Humidity. The test shall be performed at ambient pressure. When unsealed
units are being tested, precautions shall be taken to preclude condensation on and within the
unit at low temperature. For example, the chamber may be flooded with dry air or nitrogen.
Careful consideration shall also be given to the starting temperatures and temperature
transitions applied to avoid moisture condensation. A common practice is to require the
first and last half cycle to be conducted hot.
b. Temperature. Units shall be tested to temperature ranges given below and as shown in
Figure 6.3.8-4.
Qualification: 10°C beyond acceptance test temperatures or
–34 to 71°C (minimum range)
Protoqualification: 5°C beyond acceptance test temperatures or
–29 to 66°C (minimum range)
Acceptance: Maximum and minimum predicted temperatures or
–24 to 61°C (minimum range)
5 8

Figure 6.3.8-4. Unit test temperature ranges and margins.
The transition rate between hot and cold shall be at an average rate of 3°C to 5°C per minute,
and shall not be slower than 1°C per minute.
c. Duration. The minimum number of thermal cycles (TC) shall be as shown when combined
with the unit thermal vacuum (TV) test (Table 6.3-3):
Qualification: 23 TC and 4 TV cycles
Protoqualification: 16 TC and 4 TV cycles
Acceptance: 10 TC and 4 TV cycles
When an acceptance unit is insensitive to the vacuum environment (4.10.2) and only the thermal
cycle test is performed (no thermal vacuum test), the minimum number of cycles is:
Acceptance: 14 TC cycles
5 9

For Multi-Unit Module (MUM) (see 3.32), acceptance unit test cycles shall be reduced to ten thermal
cycles if four thermal vacuum cycles are performed at the MUM level. The ten unit thermal cycles
may either be unit thermal vacuum cycles or unit thermal cycles.
The last three thermal cycles shall be failure free. Units shall remain operational except during the
first and last cycle, where a minimum one-half hour is required between unit turn-off and turn-on for
stabilization. Units may be turned off during the cold ramp on intermediate cycles to accelerate
testing.
Thermal soaks at hot and cold temperature plateaus shall be a minimum of six hours on the first and
last cycle and one hour on intermediate cycles. Hot operational soaks shall be a minimum of two
hours on the first and last cycle and a minimum of one hour (coincident with thermal soak) on inter-
mediate cycles. At cold temperatures, the unit shall be thermally stabilized before proceeding to fur-
ther testing.
Temperature stabilization is achieved when the unit baseplate is within the allowed test tolerance on
the specified test temperature, and the temperature rate of change is less than 3°C per hour. Thermal
dwells at hot and cold temperatures shall be a minimum of one hour for all cycles. For intermediate
cycles, thermal dwell may be shorter than one hour, provided analysis results verify internal temper-
ature stabilization of the unit. For units in which the dwell duration may be greater than one hour,
analysis results or test data shall be used to predict the appropriate dwell duration.
When a unit’s design precludes testing over the temperature ranges specified in paragraph 6.3.8.3.b,
the number of test cycles shall be increased to provide an equivalent level of screening effectiveness.
The relationships given below shall be used to calculate equivalent cycles for units that are subjected
to thermal cycling only or subjected to thermal vacuum and thermal cycle testing. The term ∆T is the
proposed test temperature range (in °C).
Qualification: 27(105/∆T)1.4 cycles
Protoqualification: 20(95/∆T)1.4 cycles
Acceptance: 14(85/∆T)1.4 cycles
6.3.8.3.1 Option for Use of Slice/Board Testing for Acceptance Cycles
If slice/board thermal testing is performed on flight hardware, then a credit may be taken toward
meeting unit-level thermal test requirements. To allow slice/board credit toward unit level require-
ments, the following criteria shall be satisfied:
a. Slices/boards shall be powered on and monitored during testing.
b. All slices, boards, and cards in a unit shall be tested in the same manner. Temperature levels
(average values at the same relative locations) shall envelope (hot and cold) those at the unit
level.
6 0

c. Performance and functional testing, as appropriate, shall be conducted during the first and last
cycles of slice/board thermal tests at hot and cold temperature plateaus. Perceptive parame-
ters shall be monitored during all other phases of the tests.
d. Slice/board thermal test plans and procedures shall be documented in a manner similar to
unit-level thermal testing.
e. Slice/board thermal test results shall be documented and approved by the customer. The
reports shall address all anomalies, failures, corrective actions, and observations found during
the test.
f. An assessment shall be made on all items or aspects of the unit not subjected to slice/board
thermal testing (interfaces, connecting cables, subassemblies not mounted to boards, parts
mounted to chassis walls or base plate, etc.). This assessment shall consider the integrity and
robustness in meeting unit level design and performance requirements as these items are
exposed to fewer unit cycles when the slice/board thermal test credit is taken.
If the above criteria are satisfied and approved by the customer, unit-level test credit shall be applied
to the thermal cycle test and burn-in test requirements (not thermal vacuum test) in that the required
number of unit-level thermal cycles and the burn-in test duration shall be reduced. The maximum
cycle credit given for slice/board level thermal testing shall be half the required number of unit ther-
mal cycles. For example, an acceptance unit requiring 14 thermal test cycles (ten thermal cycles and
four thermal vacuum cycles) may be given a seven-cycle maximum credit for slice/board level test-
ing, and the unit thermal cycle test program is modified to three thermal test cycles and four thermal
vacuum test cycles.
When slice/board temperature ranges differ from required unit level test ranges, the relationships pro-
vided in 6.3.8.3c may be used to compute equivalent test cycles. The slice/board with the fewest
number of equivalent unit-level cycles shall be used for determining the cycle credit to be taken. For
example, a three-slice unit with three, seven, and ten equivalent unit-level cycles shall be given a
three cycle credit in reducing unit cycle requirements. Likewise for the burn-in duration credit, the
slice/board with the minimum equivalent thermal test hours shall be used for determining the unit-
level burn-in credit. Duration hours count only when the slice/board is powered on in the slice/board
test. Unlike the thermal cycle credit, the minimum equivalent duration slice/board testing that meets
the above criteria may be sufficient to eliminate the need for any unit-level burn-in testing when the
number of accrued hours in slice/board testing meets or exceeds the number of hours required at the
unit level and when all critical unit hardware has been subjected to adequate slice/board testing.
Slice/board level thermal testing shall not eliminate or reduce other unit thermal requirements (e.g.,
failure-free cycles, survival demonstration, dwell times, etc.) to be demonstrated in unit thermal tests.
6.3.8.3.2 Option for Two Tier Testing
When a unit’s performance temperature limits do not comply with Figure 6.3.8-4 temperature ranges,
but operational temperature limits do, a two-tier thermal test approach may be adopted whereby the
unit demonstrates operational requirements at the minimum temperature range shown in Figure 6.3.8-
4 and performance requirements at a narrower temperature range. The two tier test approach is
described in A.1.2.
6 1

6.3.8.4 Burn-in for Electrical and Electronic Units
For acceptance and protoqualification testing, units shall be “burned in” to detect latent infant mortal-
ity defects. During burn-in, the test unit shall be powered on, and key parameters monitored and
trended. The duration of burn-in is such that the combined duration of unit thermal cycling, unit
thermal vacuum, and the additional burn-in testing shall be at least 200 hours. The durations of the
thermal cycle and thermal vacuum test accrue toward the 200-hour duration requirement. Perfor-
mance tests shall be performed prior to and following the burn-in test. The test is performed with the
unit temperature either cycled between the acceptance temperature limits or elevated at the
acceptance hot temperature. Testing may be performed at ambient pressure as a continuation of the
unit thermal cycle test. The last 100 hours of operation shall be failure free. For burn-in tests of less
than 100 hours in duration, the test shall be failure free. For units with internal redundancy, the oper-
ating hours shall consist of at least 100 hours for each redundancy with a minimum total of 200 oper-
ating hours. The last 50 hours of each circuit (primary and redundant) shall be failure free.
6.3.8.5 Thermal Cycle Testing for Units Other Than Electrical and Electronic
When a thermal cycle test is performed for non-electrical and non-electronic units (see Table 6.3-2),
the purpose and test description are similar to that described for electrical and electronic units. The
primary difference is in the test levels and duration.
a. Pressure and Humidity. Same as 6.3.8.3.a.
b. Temperature. Units shall be tested to the ranges given below:
Qualification: 10°C beyond acceptance test temperatures
Protoqualification: 5°C beyond acceptance test temperatures
Acceptance: Maximum and minimum predicted temperatures
c. Durations. The minimum number of thermal cycles shall be:
Qualification: 6 cycles
Protoqualification: 3 cycles
Acceptance: 1 cycle
These cycles can be accrued as a combination of unit thermal cycle and unit thermal vacuum
tests (6.3.9.5).
The temperature transition rate shall be the same as stated for electrical and electronic units, but shall
also demonstrate a transition rate representative of flight conditions, if practical. The thermal stabili-
zation and thermal dwell durations shall be the same as specified for electronic and electrical units.
Thermal soak durations shall be specified for the unit to achieve thorough performance verification.
Unit performance shall be demonstrated at the required test temperatures. These requirements are
specified in 6.3.8.3.
6 2

The unit shall demonstrate survival capability requirements after exposure to survival hot and survival
cold temperatures (see 3.55) on the first cycle. Survival cycle requirements shall be identical to those
stated for electronic and electrical units.
6.3.9 Unit Thermal Vacuum Test
6.3.9.1 Purpose
The thermal vacuum test demonstrates performance and survivability over combined thermal and
vacuum conditions. The qualification thermal vacuum test demonstrates the ability of the unit to
perform to specification limits in the qualification environment and to endure the thermal vacuum
testing imposed on flight units during acceptance testing. It also serves to verify the unit thermal
design. The acceptance thermal vacuum test detects material and workmanship defects and proves
flightworthiness of the unit. Criteria for exemptions to unit acceptance thermal vacuum testing are
given in 4.10.2.
6.3.9.2 Test Description for Electrical and Electronic Units
The unit under test shall be mounted in a vacuum chamber on a thermally controlled heat sink or in a
manner similar to its actual installation in the vehicle. The unit surface finishes, which affect radia-
tive heat transfer or contact conductance, shall be thermally equivalent to those on the flight units.
For units designed to reject their waste heat through the baseplate, a control temperature sensor shall
be attached to either the unit baseplate or the heat sink. The location shall be chosen to correspond as
closely as possible to the temperature limits used in the vehicle thermal design analysis or applicable
unit-to-vehicle interface criteria. For components cooled primarily by radiation, a representative
location on the unit case shall similarly be chosen. The unit heat transfer to the thermally controlled
heat sink and the radiation heat transfer to the environment shall be controlled to the same proportions
as calculated for the flight environment.
The chamber pressure shall be reduced to the required vacuum conditions. Units that are required to
operate during ascent shall be operating and monitored for arcing and corona during the reduction of
pressure to the specified lowest levels and during the early phase of vacuum operation. Units that do
not operate during launch shall have electrical power applied after the test pressure level has been
reached.
A thermal cycle begins with the conductive or radiant sources and sinks at ambient temperature.
With the unit operating (power on), the unit shall be tested while subjected to the temperature profiles
in Figures 6.3.8-1, 6.3.8-2, and 6.3.8-3, or as appropriate, for first, intermediate and last cycle testing,
respectively. The unit temperature shall be raised to the specified hot temperature and maintained for
thermal dwell to ensure the unit internal temperature has stabilized. On the first cycle, survival
requirements shall be demonstrated at the unit’s survival hot and cold limits as described for unit
thermal cycle testing (6.3.8.2). On the first and last cycles, the unit shall be turned off, then hot-
started and performance tested. Following the thermal soak and with the unit operating, the unit tem-
perature shall be reduced to the specified cold temperature. To aid in reaching the cold temperature,
the unit may be powered off at the completion of hot performance testing. After the unit temperature
has reached the specified cold temperature, the unit shall be turned off (if not previously turned off
during the transition) until the internal temperature stabilizes through the thermal dwell, and then cold
started and performance tested. The unit shall be maintained at the cold temperature until the end of
6 3

the thermal soak. The temperature of the sinks shall then be raised to ambient conditions. This con-
stitutes one complete thermal cycle.
On the first cycle, the unit shall demonstrate survival capability requirements at its survival hot and
survival cold temperatures (see 3.55). Following a thermal dwell at the survival temperature, the unit
shall be maintained at the survival temperature for a minimum of one hour. For a unit with a hot or
cold turn-on temperature requirement different from its operational value, turn-on capability at the
turn-on temperature shall be demonstrated on the first cycle.
Compliance to specified performance requirements shall be required over acceptance, protoqualifi-
cation, and qualification temperature ranges. Performance tests shall be conducted after electrical and
electronic unit temperatures have stabilized (see 3.57) at the hot and cold temperatures during the first
and last cycle, and at ambient temperature prior to, and following, the test. Functional tests shall be
performed at hot and cold temperature plateaus on intermediate cycles. These tests shall include:
a. During performance and functional tests, the unit shall be cycled through a sufficient number
of operational modes to fully characterize the performance and functionality of the unit.
b. Performance tests shall test all paths, including continuity, with functional testing as a subset
of performance testing that focuses on critical, primary, and redundant paths to check
functionality.
c. During the test, perceptive parameters shall be monitored for failures, degradation trends, and
intermittent behavior.
d. All electrical circuits and all paths shall be verified for circuit performance and continuity.
e. For units with internal redundancy, performance testing, functional testing, and hot and cold
starts shall be demonstrated on primary and redundant circuits and paths.
f. During temperature transitions, the unit shall be powered on and monitored, except as noted,
and the health of the unit shall be monitored and key parameters trended. Any performance
of the unit required during temperature transitions shall be tested during the test transitions.
g. Any performance of the unit required during temperature transitions shall be tested during the
test transitions.
6.3.9.3 Test Levels and Durations for Electrical and Electronic Units
a. Pressure. The time for reduction of chamber pressure from ambient to 20 Pa (0.15 Torr)
shall be at least ten minutes to allow sufficient time in the region of critical pressure for units
required to operate during ascent. The pressure shall be further reduced from 20 Pa for oper-
ating equipment, or from atmospheric for equipment that does not operate during ascent, to
13.3 mPa (10–4 Torr) at a rate that simulates the ascent profile to the extent practical. For
launch vehicle units, the vacuum pressure test shall be modified to reflect an altitude consis-
tent with the maximum service altitude and duration consistent with maximum time at
altitude.
b. Temperature. Electrical and electronic units shall be tested to temperature ranges given
below and as shown in Figure 6.3.8-4
6 4

Qualification: 10°C beyond acceptance test temperatures or –34 to 71°C (minimum
range)
Protoqualification: 5°C beyond acceptance test temperatures or –29 to 66°C
(minimum range)
Acceptance: Maximum and minimum predicted temperatures or –24 to 61°C
(minimum range)
The transitions between hot and cold shall be at an average rate greater than 1°C per minute.
c. Duration. Electrical and electronic units shall have the following minimum number of ther-
mal vacuum cycles (with thermal cycling performed):
Qualification: 4 TV cycles
Protoqualification: 4 TV cycles
Acceptance: 4 TV cycles
When performing thermal vacuum testing only, the minimum number of cycles shall be:
Qualification: 27 TV cycles
Protoqualification: 20 TV cycles
Acceptance: 14 TV cycles
Units shall remain operational except during the first and last cycles, where a minimum one-half hour
is required between unit turn-off and turn-on for stabilization. Units may be turned off during the
cold ramp on intermediate cycles to accelerate testing, but the unit shall be thermally stabilized after
being turned on before proceeding to functional or performance testing. When performing thermal
vacuum testing only, the last three thermal cycles shall be failure free.
Thermal soaks at hot and cold temperature plateaus shall be a minimum of six hours on the first and
last cycle and one hour on intermediate cycles. Hot operational soaks shall be a minimum of two
hours on the first and last cycle and a minimum of one hour (coincident with thermal soak) on inter-
mediate cycles.
Temperature stabilization is achieved when the unit baseplate is within the allowed test tolerance on
the specified test temperature, and the temperature rate of change is less than 3°C per hour. Thermal
dwells at hot and cold temperatures shall be a minimum of four hours for all cycles. Thermal dwell
durations greater than four hours may be necessary to ensure that internal locations have reached the
test temperature. In such cases, either thermal design analysis results or test measurements of internal
unit components shall be used to predict an appropriate dwell time. Thermal dwell durations of less
than four hours may be used when analysis results or test measurements indicate that such a change is
appropriate.
6 5

When a unit’s design precludes testing over the temperature ranges specified in 6.3.9.3.b, the number
of test cycles shall be increased to provide an equivalent level of screening effectiveness. The rela-
tionships given below shall be used to calculate equivalent cycles for units that are subjected to ther-
mal vacuum only or subjected to thermal vacuum and thermal cycle testing. The term ∆T is the pro-
posed test temperature range (in °C).
Qualification: 27(105/∆T)1.4 cycles
Protoqualification: 20(95/∆T)1.4 cycles
Acceptance: 14(85/∆T)1.4 cycles
6.3.9.3.1 Option for Two-Tier Testing
When a unit’s performance temperature limits do not comply with Figure 6.3.8-4 temperature ranges,
but operational temperature limits do, a two-tier thermal test approach may be adopted whereby the
unit demonstrates operational requirements at the minimum temperature range shown above and per-
formance requirements at a narrower temperature range. The two-tier test approach is described in
A.1.2.
6.3.9.4 Burn-In for Electrical and Electronic Units
For acceptance and protoqualification testing, units shall be “burned in” to detect latent infant mortal-
ity defects. During burn-in, the test unit shall be powered on, and key parameters monitored and
trended. The duration of burn-in is such that the combined duration of unit thermal cycling, unit
thermal vacuum, and the additional burn-in testing shall be at least 200 hours. The durations of the
thermal cycle and thermal vacuum test accrue toward the 200-hour duration requirement. Perfor-
mance tests shall be performed prior to and following the burn-in test. The test is performed with the
unit temperature either cycled between the acceptance temperature limits or elevated at the
acceptance hot temperature. Testing may be performed at ambient pressure as a continuation of the
unit thermal cycle test. The last 100 hours of operation shall be failure free. For burn-in tests of less
than 100 hours in duration, the test shall be failure free. For units with internal redundancy, the oper-
ating hours shall consist of at least 100 hours for each redundancy with a minimum total of 200 oper-
ating hours. The last 50 hours of each circuit (primary and redundant) shall be failure free.
6.3.9.5 Thermal Vacuum Testing for Units Other Than Electrical and Electronic
When a thermal vacuum test is performed for non-electrical and non-electronic units, the purpose and
test description are not significantly different from that described for electrical and electronic units.
The primary difference is in the test levels and duration. Non-electrical and non-electronic units are
tested to temperature ranges that have the same thermal margins, but without the required temperature
ranges:
a. Temperature. Units shall be tested to the ranges given below:
Qualification: 10°C beyond acceptance test temperatures
6 6

Protoqualification: 5°C beyond acceptance test temperatures
Acceptance: Maximum and minimum predicted temperatures
b. Duration. The minimum number of thermal cycles shall be:
Qualification: 6 TV cycles
Protoqualification: 3 TV cycles
Acceptance: 1 TV cycle
These cycles can be accrued as a combination of unit thermal cycle (6.3.8.5) and unit
thermal vacuum tests.
The test shall demonstrate a transition rate representative of flight conditions. The thermal stabiliza-
tion and thermal dwell durations shall be the same as specified for electronic and electrical units.
Thermal soak durations shall be specified for the unit to achieve thorough performance verification.
Unit performance shall be demonstrated at the required test temperatures. These requirements are
specified in 6.3.9.3.
The unit shall demonstrate survival capability requirements at its survival hot and survival cold tem-
peratures (see 3.55) on the first cycle. Survival cycle requirements shall be identical to those stated
for electronic and electrical units.
For moving mechanical assemblies, performance parameters (such as current draw, resistance torque
or force, actuation time, velocity, or acceleration) shall be monitored. Where practical, force or
torque margins shall be determined on moving mechanical assemblies at the temperature extremes.
Where this is not practical, minimum acceptable force or torque margin shall be demonstrated.
Compatibility with operational fluids shall be verified at test temperature extremes for valves, propul-
sion units, and other units, as appropriate.
6.3.10 Unit Climatic Tests
6.3.10.1 Purpose
These tests demonstrate that the unit is capable of surviving exposure to various climatic conditions
without excessive degradation, or operating during exposure, as applicable. Exposure conditions
include those imposed upon the unit during fabrication, test, shipment, storage, preparation for
launch, launch itself, and reentry, if applicable. These can include, but not be limited to, such condi-
tions as humidity, sand and dust, rain, salt fog, and explosive atmosphere. Tests shall conform to the
methods given in Reference 1 when applicable. Degradation due to fungus, ozone, and sunshine shall
be verified by design and material selection.
6 7

It is the intent that environmental design of flight hardware not be driven by terrestrial natural envi-
ronments. To the greatest extent feasible, the flight hardware shall be protected from the potentially
degrading effects of extreme terrestrial natural environments by procedural controls and special sup-
port equipment. Only those environments that cannot be controlled need be considered in the design
and testing.
6.3.10.2 Humidity Test, Unit Qualification
6.3.10.2.1 Purpose
The humidity test demonstrates that the unit is capable of surviving or operating in, if applicable,
warm humid environments. In the cases where exposure is controlled throughout the life cycle to
conditions with less than 55% relative humidity, and the temperature changes do not create conditions
where condensation occurs on the hardware, then verification by test is not required.
6.3.10.2.2 Test Description and Levels
For units exposed to unprotected ambient conditions, the humidity test shall conform to the method
507.5 in Reference 1. For units located in protected, but uncontrolled environments, the unit shall be
installed in a humidity chamber and subjected to the following conditions (time line illustrated in
Figure 6.3.10-1):
Figure 6.3.10-1. Humidity test time line.
6 8

a. Pretest Conditions. Chamber temperature shall be at room ambient conditions with uncon-
trolled humidity.
b. Cycle 1. The temperature shall be increased to +35°C over a one-hour period; then the
humidity shall be increased to not less than 95% over a one-hour period with the temperature
maintained at +35°C. These conditions shall be maintained for two hours. The temperature
shall then be reduced to +2°C over a two-hour period with the relative humidity stabilized at
not less than 95%. These conditions shall be maintained for two hours.
c. Cycle 2. Cycle 1 shall be repeated except that the temperature shall be increased from +2°C
to +35oC over a two-hour period; moisture is not added to the chamber until +35°C is
reached.
d. Cycle 3. The chamber temperature shall be increased to +35°C over a two-hour period with-
out adding any moisture to the chamber. The test unit shall then be dried with air at room
temperature and 50% maximum relative humidity by blowing air through the chamber for six
hours. The volume of air used per minute shall be equal to one to three times the test cham-
ber volume. A suitable container may be used in place of the test chamber for drying the test
unit.
e. Cycle 4. If it had been removed, the unit shall be placed back in the test chamber, the temp-
erature increased to +35°C, and the relative humidity increased to 90% over a one-hour per-
iod; and these conditions shall be maintained for at least one hour. The temperature shall
then be reduced to +2°C over a one-hour period with the relative humidity stabilized at 90%;
and these conditions shall be maintained for at least one hour. A drying cycle shall follow
(see Cycle 3).
6.3.10.2.3 Supplementary Requirements
The unit shall be functionally tested prior to the test and at the end of Cycle 3 (within two hours after
the drying) and visually inspected for deterioration or damage. The unit shall be functionally tested
during the Cycle 4 periods of stability, after the one-hour period to reach +35°C and 90% relative
humidity, and again after the one-hour period to reach the +2°C and 90% relative humidity.
6.3.10.3 Sand and Dust Test, Unit Qualification
6.3.10.3.1 Purpose
The sand and dust test is conducted to determine the resistance of units to blowing fine sand and dust
particles. This test shall not be required for units protected from sand and dust by contamination
control, protective shipping and storage containers, or covers. However, in those cases, rain testing
demonstrating the adequacy of the protective shelters, shipping and storage containers, or covers, as
applicable, may be required instead of a test of the unit itself.
6.3.10.3.2 Test Description
The test requirements for the sand and dust test shall conform to the method given in Reference 1.
6 9

6.3.10.4 Rain Test, Unit Qualification
6.3.10.4.1 Purpose
The rain test shall be conducted to determine the resistance of units to rain. Units protected from rain
by protective shelters, shipping and storage containers, or covers, shall not require verification by test.
6.3.10.4.2 Test Description
Buildup of the unit, shelter, container, or the cover being tested shall be representative of the actual
fielded configuration, without any duct tape or temporary sealants. The initial temperature difference
between the test item and the spray water shall be a minimum of 10°C. For temperature-controlled
containers, the temperature difference between the test item and the spray water shall at least be that
between the maximum control temperature and the coldest rain condition in the field. Nozzles used
shall produce a square spray pattern or other overlapping pattern (for maximum surface coverage) and
droplet size predominantly in the 2 to 4.5 mm range at approximately 375 kPa gage pressure (40
psig). At least one nozzle shall be used for each approximately 0.5 m2 (6 ft2) of surface area, and
each nozzle shall be positioned at 0.5 m (20 in.) from the test surface. All exposed faces shall be
sprayed for at least 40 minutes. The unit under test interior shall be inspected for water penetration at
the end of each 40-minute exposure.
6.3.10.5 Salt Fog Test, Unit Qualification
6.3.10.5.1 Purpose
The salt fog test is used to demonstrate the resistance of the unit to the effects of a salt spray atmo-
sphere. The salt fog test is not required if the flight hardware is protected against the salt fog envi-
ronment by suitable preservation means and protective shipping and storage containers.
6.3.10.5.2 Test Description
The requirements for the salt fog test shall conform to the method given in Reference 1.
6.3.10.6 Explosive Atmosphere Test, Unit Qualification
6.3.10.6.1 Purpose
Where applicable, devices operating in explosive atmospheric conditions need to be proven incapable
of igniting a fuel-air mixture of concern.
6.3.10.6.2 Test Description
The test requirements for the explosive atmosphere test shall conform to the method given in
Reference 1.
6.3.11 Unit Static Load Test
6.3.11.1 Purpose
The structural static load test demonstrates the adequacy of the structural components to meet
requirements of strength and stiffness, with the desired test factors, when subjected to simulated criti-
cal environments predicted to occur during its service life (such as loads, temperature, humidity, and
pressure).
7 0

6.3.11.2 Test Description
The interface between the test article and the test fixture shall provide flight-like boundary conditions
including stiffness. The test fixture shall allow for the proper sequencing or simultaneous application
of all load cases. When prior loading histories affect the structural adequacy of the test article, these
shall be included in the test requirements. If more than one design load condition is to be applied to
the same test specimen, a method of sequential load application shall be developed by which each
condition may, in turn, be tested to progressively higher load levels.
Measurements of strains and deformations shall be recorded for all static load cases. Strain, load, and
deformation shall be measured before and during loading, after removal of the loads, and at several
intermediate levels for post-test diagnostic purposes. The test conditions shall encompass the extreme
predicted combined effects of acceleration, vibration, pressure, preloads, and temperature. These
effects can be simulated in the test conditions as long as the design margins for all failure modes are
enveloped by the test. For example, temperature effects, such as material strength degradation and
additive thermal stresses, can often be accounted for by increasing mechanical loads.
6.3.11.3 Test Levels and Duration
Qualification:
a. Level. Unless otherwise specified, the load level for the static load test is 1.4 times limit
load (3.21) for manned systems and 1.25 times limit load for unmanned systems.
b. Temperature. Critical flight temperature and load combinations shall be simulated or
taken into account by modification of the applied mechanical loads.
c. Duration. The dwell time at each load level shall be sufficient to achieve stable structural
response and record test data such as strain, load, displacement, and temperature.
Protoqualification:
Same as qualification except the load level for static test is 1.25 limit load (3.21) for manned
and unmanned systems.
Acceptance:
A unit proof load test (see 3.42) shall be conducted for all structural units made of composite
materials or having adhesively bonded parts. The proof load test is intended to detect mate-
rial, process, and workmanship defects that could lead to structural failure.
a. Level. Unless otherwise specified, the proof load for flight items shall be 1.1 times limit load
(see 3.21).
b. Duration. The dwell time at each load level shall be sufficient to achieve stable structural
response and record test data.
Proof testing may be deleted if the following conditions are met:
7 1

a. Units are manufactured using established and controlled processes
b. Nondestructive inspection methods, together with well-established accept/reject criteria,
are used to verify the workmanship of each unit. The selected methods have been
demonstrated to be effective for identifying critical flaws in units of identical or similar
geometry, construction, and materials.
c. Qualification/Protoqualification on a similar unit has been successfully completed
d. Mechanical properties of each component material are verified by tag end or witness tests
and whose data are in-family (2-sigma) to heritage data
e. Approval obtained from customer
6.3.11.3.1 Test Success Criteria
Qualification: The unit shall withstand the qualification loads without rupture or
collapse. There shall be no material gross yielding or detrimental
deformation at 1.1x limit loads.
Protoqualification: The unit shall withstand the applied loads without material gross yielding
or detrimental deformation.
Acceptance: The unit shall withstand the applied loads without material gross yielding
or detrimental deformation
6.3.11.4 Supplementary Requirements
For fracture-critical metallic parts, proof tests shall be conducted when non-destructive evaluation is
not sufficient to determine the maximum initial crack sizes used in the damage tolerance (safe-life)
analyses or tests. The required proof test load level shall be determined based on fracture mechanics
calculations. Structural test requirements are specified in References 6 and 11. Qualification or
protoqualification testing shall be required for all composite and bonded primary structures.
6.3.12 Unit Pressure Test
6.3.12.1 Purpose
The pressure test verifies adequate margin that structural failure does not occur before the design
burst pressure is reached, or excessive deformation does not occur at the maximum expected
operating pressure, MEOP (see 3.24). Table 6.3.12-2 provides minimum design burst pressures for
pressurized vessels and hardware.
For solid rocket motor cases used in expendable launch vehicles, Reference 19 specifies applicable
requirements.
6.3.12.2 Test Description
a. Proof Pressure Test. For items such as pressurized structures, vessels, and pressure compo-
nents, a proof test with a minimum of one cycle of proof pressure shall be conducted. Evi-
dence of leakage, a permanent set, or distortion that exceeds a drawing tolerance or failure of
7 2

any kind shall constitute failure to pass the test. This test shall be performed for qualification
and acceptance testing.
b. Qualification Test. The qualification test procedures consist of cyclic testing followed by
one additional cycle at burst pressure, and are described below. Requirements for application
of external loads in combination with internal pressures during testing shall be evaluated
based on the relative magnitude and on the destabilizing effect of stresses due to the external
load.
Pressure Cycle Test. For pressurized structures and pressure vessels, a pressure
cycle test shall be conducted. If limit combined tensile stresses are enveloped by
the test pressure stress, the application of external load is not required. Table
6.3.12-1 provides a summary of unit test requirements.
Burst Test. For pressurized structures and vessels, after demonstrating no burst
at the design burst pressure, the pressure shall be increased to actual burst of the
test unit, and the actual burst pressure shall be recorded.
c. Exception to Tests. For special pressurized equipment (see 3.38), such as silver-zinc
batteries that contain a pressure relief mechanism, proof test of the pressure release
mechanism shall be performed on all flight units. For space vehicle batteries using a special
pressurized equipment design without a pressure release mechanism, such as nickel-hydrogen
batteries, proof testing shall be performed at the vessel level for each vessel in a flight
battery. See Reference 4 for design and test requirements.
Table 6.3.12-1. Unit Pressure Cycle and Burst Test Requirements
Hardware Type Pressure Cycles Burst Pressure
Pressurized Structures Cycle at 1.0 times MEOP for 4 times pre- 1.25 x MEOP
dicted number of service life cycles in
sequence, including proof test
Metallic Pressure Vessels Cycle at 1.0 times MEOP for 4 times pre- 1.5 x MEOP
dicted number of service life cycles in
sequence (50 cycles minimum)
or
Cycle at 1.5 times MEOP for 2 times pre-
dicted number of service life cycle in
sequence. (50 cycles minimum)
Composite Overwrapped Pressure Cycle for 4 times service life cycles, including 1.5 x MEOP
Vessels with Metal Liners proof tests (50 cycles minimum)
Table 6.3.12-2. Minimum Design Burst Pressure Requirements
Pressurized Hardware Item Type Minimum Design Burst Pressure
Pressurized Structures 1.25 x MEOP
Metallic Pressure Vessels, Cryostats, Battery Cases and Sealed Containers 1.5 x MEOP
Composite Overwrapped Pressure Vessels with Metal Liners 1.5 x MEOP
Lines and Fittings with Diameters Equal to or Greater Than 1.5 in. 2.5 x MEOP
Heat Pipes, Valves, Regulators, Accumulators, and others Pressure 2.5 x MEOP
Components
Fluid Return Section 3.0 x MEOP
Lines and Fittings with Diameters Less Than 1.5 in. 4.0 x MEOP
Fluid Return Hose 5.0 x MEOP
7 3

| Hardware Type | Pressure Cycles | Burst Pressure |
| --- | --- | --- |
| Pressurized Structures | Cycle at 1.0 times MEOP for 4 times pre- dicted number of service life cycles in sequence, including proof test | 1.25 x MEOP |
| Metallic Pressure Vessels | Cycle at 1.0 times MEOP for 4 times pre- dicted number of service life cycles in sequence (50 cycles minimum) or Cycle at 1.5 times MEOP for 2 times pre- dicted number of service life cycle in sequence. (50 cycles minimum) | 1.5 x MEOP |
| Composite Overwrapped Pressure Vessels with Metal Liners | Cycle for 4 times service life cycles, including proof tests (50 cycles minimum) | 1.5 x MEOP |

| Pressurized Hardware Item Type Minimum Design Burst Pressure |
| --- |
| Pressurized Structures 1.25 x MEOP |
| Metallic Pressure Vessels, Cryostats, Battery Cases and Sealed Containers 1.5 x MEOP |
| Composite Overwrapped Pressure Vessels with Metal Liners 1.5 x MEOP |
| Lines and Fittings with Diameters Equal to or Greater Than 1.5 in. 2.5 x MEOP |
| Heat Pipes, Valves, Regulators, Accumulators, and others Pressure 2.5 x MEOP Components |
| Fluid Return Section 3.0 x MEOP |
| Lines and Fittings with Diameters Less Than 1.5 in. 4.0 x MEOP |
| Fluid Return Hose 5.0 x MEOP |

6.3.12.3 Test Levels and Durations
a. Temperature and Humidity. The test temperature and humidity conditions shall be consis-
tent with the critical-use temperature and humidity. As an alternative, tests may be conducted
at ambient conditions if the test pressures are suitably adjusted to account for temperature and
humidity effects on material strength and fracture toughness.
b. Proof Pressure. Unless otherwise specified, the minimum proof pressure for pressurized
structures shall be 1.1 times the MEOP. For pressure vessels, and other pressure components
such as lines and fittings, the minimum proof pressure shall comply with the requirements
specified in References 4 and 5, as appropriate. The hold time for pressure vessels shall
comply with the requirements specified in References 4 and 5, as appropriate. For other
pressure components, the pressure shall be maintained for a time just sufficient to assure that
the proper pressure was achieved.
c. Pressure Cycle. Unless otherwise specified, the peak pressure for pressurized structures
shall equal the MEOP during each cycle, and the number of cycles shall be four times the
predicted number of operating cycles. For pressure vessels, the test shall comply with the
requirements specified in References 4 and 5, as appropriate.
d. Burst Pressure. For pressurized structures, vessels and components, the minimum design
burst pressure and duration shall comply with References 4 and 5, as appropriate.
6.3.12.4 Supplementary Requirements
Applicable safety standards shall be followed in conducting all tests. Unless otherwise specified,
the qualification testing of metallic pressure vessels shall include a demonstration of a leak-
before-burst (LBB) failure mode using pre-flawed specimens as specified in Reference 4. The
LBB pressure test may be omitted if available material data are directly applicable to be used for
an analytical demonstration of the leak-before-burst failure mode.
For composite over-wrapped pressure vessels with metallic liners, the LBB requirements speci-
fied in Reference 5 shall be met.
For perforated sandwich structures, when venting tests cannot simulate the flight depressurization
rate, tests shall be conducted to verify the unit has sufficient strength to withstand internal
pressure.6.3.13 Unit Electromagnetic Compatibility (EMC) Test
6.3.13 Unit Electromagnetic Compatibility (EMC) Test
6.3.13.1 Purpose
The electromagnetic compatibility test shall demonstrate that the electromagnetic interference char-
acteristics (emission and susceptibility) of the unit, under normal operating conditions, do not result
in malfunction of the unit. It also demonstrates that the unit does not emit, radiate, or conduct inter-
ference, which could result in malfunction of other units.
7 4

6.3.13.2 Test Description
The test shall be conducted in accordance with the requirements of Reference 21. The intent of
testing at the lowest level possible shall be followed. This means that all tests shall be conducted at
the unit level to improve the chance of passing the subsystem and vehicle level tests. Radiated
emissions shall be performed on all units capable of generating emissions. Acceptance tests shall be
performed when there is less than 12 dB qualification margin, or the radiated emissions requirement
is more stringent than 10 dBuV/m, or the units have a passive intermodulation requirement. Radiated
emission acceptance tests can be deferred to the subsystem or functional module level.
The EMC margin is to be incorporated into the test levels. Qualification margins of 6 dB are accept-
able if the combined test uncertainty, part variation, part degradation at end-of-life, and workmanship
variation is less than 6 dB. Electroexplosive devices and bridge wires have a 20 dB margin require-
ment below the DC no-fire value and a 6 dB margin requirement below the RF no-fire value.
6.3.13.3 Test Levels and Duration
The test levels shall be as follows:
Qualification: 12 dB
Protoqualification: 6 dB
Acceptance: 6 dB
The test duration shall be 20 minutes at each space vehicle transmitter frequency for radiated
susceptibility. Otherwise, the duration is the greater of three seconds or the unit response time for
susceptibility requirements and 15 ms duration for emission requirements.
6.3.14 Unit Life Test
6.3.14.1 Purpose
The life test applies to units which may have a wear-out, drift, or fatigue-type failure mode, or per-
formance degradation in the operational environment. The test demonstrates that the units have the
capability to perform within specification limits for the maximum duration or cycles of operation
during repeated ground testing and flight. The unit life test is an environmental test and not to be
confused with pressure cycle test, which is covered in 6.3.12.3.
6.3.14.2 Test Description
One or more units shall be operated under conditions that simulate their service conditions. Service
conditions define the operational environment in which the unit is expected to operate. As such, it is
reasonable to demonstrate the condition when the unit is in an active and operational mode. For solar
cell devices and unit life testing, the test conditions are to simulate the as-used powered-on or current
loaded operational condition. These conditions shall be selected for consistency with end-use
requirements and the significant life characteristics of the particular unit. Typical environments are
ambient, thermal, thermal vacuum pressure, vibration and radiation. The test shall be designed to
demonstrate the ability of the unit to withstand the maximum operating time and/or the maximum
7 5

number of operational cycles predicted during its service life (including manufacturing assembly and
test) with a suitable margin. Accelerated life testing is permitted provided the acceleration method is
valid for the expected operational conditions.
6.3.14.3 Test Levels and Durations
a. Pressure. Ambient pressure shall be used except where degradation due to a vacuum
environment may be anticipated. Examples include unsealed units such as bearings, coaxial
cables routed across rotating joints, or any other friction-prone device. For these cases, a
pressure of 13.3 mPa (10–4 Torr) or less shall be used.
b. Environmental Levels. The maximum expected environmental levels shall be used. Higher
levels may be used to accelerate the life testing if the resulting increase in the rate of degra-
dation is well established and that unrealistic failure modes are not introduced.
c. Duration. The total operating time or number of operational cycles shall be at least two
times that predicted during the service life, including ground testing, in order to demonstrate
an adequate margin. For a structural component having a fatigue-type failure mode that has
not been subjected to a vibration qualification test, the test duration shall be at least four times
the specified service life.
d. Functional Duty Cycle. Complete functional tests shall be conducted before the test begins
and after completion of the test. During the life test, functional tests shall be conducted in
sufficient detail, and at sufficiently short intervals to establish trends.
6.3.14.4 Supplementary Requirements
Life testing of moving mechanical assemblies (MMA) shall be performed according to Reference 7.
Life testing of batteries shall be performed according to Reference 20.
For statistically based life tests, the duration is dependent upon the number of samples, confidence,
and reliability to be demonstrated. The duration of the life test shall assure with high confidence that
the unit does not wear out and/or unacceptably degrade during its service life.
Critical areas of parts that may be subject to fatigue failure shall be inspected to determine their integ-
rity. Life testing is necessary for pressure vessels using bellows or other flexible fluid devices or
lines. Life testing on a lot basis is necessary for silver-zinc batteries to verify capacity and voltage
response at the end of wet stand life for at least one charge cycle.
7 6

7. Subsystem Test Requirements
Subsystems shall be tested to reduce risk when system testing cannot verify subsystem performance.
For example, structural testing and mode survey should be performed at subsystem level to provide
early verification.
7.1 Requirements
Subsystem tests shall be conducted on subsystems for any of the following purposes:
a. To verify the design and performance and demonstrate that those subsystems
subjected to environmental acceptance tests perform to specification (Table 7.3-2).
Table 7.3-1 summarizes qualification and protoqualification testing performed to
verify design margins.
b. To provide a more perceptive test versus other levels of testing. A summary of sub-
system test level margins and durations is shown in Table 7.3-3.
Table 7.3-1. Subsystem Qualification and Protoqualification Test Summary
Suggested Payload Payload/ Multi-Unit
Test Reference Sequence Fairing Structure Bus Instrument Module
Inspection (1) 4.6 1, 11 R R R R R
Performance(1) 7.3.1 2–10 R R R R R
Static Load 7.3.2 3 R R R R R
Pressure and Leak 7.3.3 4 ER ER ER(4) ER ER
Shock 7.3.6 7 --(5) ER ER ER ER
Random Vibration 7.3.4 5 R ER ER ER ER
or
Acoustic 7.3.5
Thermal Vacuum 7.3.7 6 -- ER ER R R
Separation and 7.3.8 8 R -- R R ER
Deployment(3)
EMC 7.3.9 9 ER(6) -- R R ER
Mode Survey 7.3.10 Any R -- R(2) R(2) ER
R Required
ER Evaluation required (see 6.3)
(1) Performance tests conducted prior to, during and following each environmental test, as
appropriate
(2) Mode survey testing is required for both if not performed at the System level
(3) Preferred at the vehicle level; if not feasible, perform test at the subsystem level
(4) Required for propulsion subsystem
(5) Performed as part of the separation and deployment test
(6) Evaluation required when active electronics installed on fairing
7 7

| Test | Reference | Suggested Sequence | Payload Fairing | Structure | Bus | Payload/ Instrument | Multi-Unit Module |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Inspection (1) | 4.6 | 1, 11 | R | R | R | R | R |
| Performance(1) | 7.3.1 | 2–10 | R | R | R | R | R |
| Static Load | 7.3.2 | 3 | R | R | R | R | R |
| Pressure and Leak | 7.3.3 | 4 | ER | ER | ER(4) | ER | ER |
| Shock | 7.3.6 | 7 | --(5) | ER | ER | ER | ER |
| Random Vibration or Acoustic | 7.3.4 7.3.5 | 5 | R | ER | ER | ER | ER |
| Thermal Vacuum | 7.3.7 | 6 | -- | ER | ER | R | R |
| Separation and Deployment(3) | 7.3.8 | 8 | R | -- | R | R | ER |
| EMC | 7.3.9 | 9 | ER(6) | -- | R | R | ER |
| Mode Survey | 7.3.10 | Any | R | -- | R(2) | R(2) | ER |

Table 7.3-2. Subsystem Acceptance Test Summary
Suggested Payload Multi-Unit
Test Reference Sequence Fairing Structure Bus Payload Module
Inspection(1) 4.6 1, 11 R R R R R
Performance(1) 7.3.1 2, 10 R R R R R
Static Load 7.3.2 3 ER(2) ER(2) ER(2) R --
Pressure and Leak 7.3.3 4 -- -- R R R
Shock 7.3.6 5 --(5) ER ER ER ER
Random Vibration 7.3.4 6
or ER -- -- ER R
Acoustic 7.3.5
Thermal Vacuum 7.3.7 7 -- -- ER R R
Separation and 7.3.8 8 ER -- R R ER
Deployment(3)
EMC(4) 7.3.9 9 -- -- ER ER --
R Required
ER Evaluation required (see 6.3)
(1) Performance tests conducted prior to, during and following each environmental test, as
appropriate
(2) Required for composite and/or bonded structures. Evaluation required for all other
structures.
(3) Preferred at the vehicle level; if not feasible, perform test at the subsystem level.
(4) Required when there is less than 12 dB qualification margin
(5) Performed as part of the separation and deployment test
Table 7.3-3. Subsystem Test Level Margins and Durations
Test Qualification Protoqualification Acceptance
Shock 1 activation of all shock-producing 1 activation of all shock- 1 activation of significant shock-
events; producing events; producing events
2 additional activations of 1 additional activation of
significant events significant events
Acoustic(1) 6 dB above acceptance for 3 min 3 dB above acceptance for 2 Envelope of MPE and minimum spectrum
min (Figure 6.3.6-1) for 1 minute
Vibration(1) 6 dB above acceptance for 3 min 3 dB above acceptance for 2 Envelope of MPE and minimum spectrum
in each of 3 axes min in each of 3 axes (Figure 8.3.7-1) for 1 min in each of 3
axes
Thermal ±10oC beyond acceptance for 8 ±5oC beyond acceptance for MPT for 4 cycles
Vacuum cycles 4 cycles
Static 1.25 times the limit load for 1.25 times the limit load for 1.1 times the limit load for bonded,
Load(2) unmanned flight or unmanned flight or composite, or sandwich structures;
1.4 times limit load for manned 1.25 times limit load for duration sufficient to record data
flight; duration sufficient to record manned flight; duration
data sufficient to record data
Pressure(3) Not applicable Not applicable 1.1 times MEOP for pressurized
structures.
1.5 times MEOP for pressurized
subsystems including only pressure
components.
1.25 times MEOP for pressurized
subsystems containing pressure vessels.
EMC 12 dB minimum; duration same as 6 dB minimum; duration 6 dB minimum; 20 min at each space
acceptance same as acceptance vehicle transmitter frequency for radiated
susceptibility
(1) See B.1.3 for subsystems with effective duration greater than 15 seconds
(2) Refer to References 6 and 11
(3) Refer to References 18 and 19
7 8

| Test | Reference | Suggested Sequence | Payload Fairing | Structure | Bus | Payload | Multi-Unit Module |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Inspection(1) | 4.6 | 1, 11 | R | R | R | R | R |
| Performance(1) | 7.3.1 | 2, 10 | R | R | R | R | R |
| Static Load | 7.3.2 | 3 | ER(2) | ER(2) | ER(2) | R | -- |
| Pressure and Leak | 7.3.3 | 4 | -- | -- | R | R | R |
| Shock | 7.3.6 | 5 | --(5) | ER | ER | ER | ER |
| Random Vibration or Acoustic | 7.3.4 7.3.5 | 6 | ER | -- | -- | ER | R |
| Thermal Vacuum | 7.3.7 | 7 | -- | -- | ER | R | R |
| Separation and Deployment(3) | 7.3.8 | 8 | ER | -- | R | R | ER |
| EMC(4) | 7.3.9 | 9 | -- | -- | ER | ER | -- |

| Test | Qualification | Protoqualification | Acceptance |
| --- | --- | --- | --- |
| Shock | 1 activation of all shock-producing events; 2 additional activations of significant events | 1 activation of all shock- producing events; 1 additional activation of significant events | 1 activation of significant shock- producing events |
| Acoustic(1) | 6 dB above acceptance for 3 min | 3 dB above acceptance for 2 min | Envelope of MPE and minimum spectrum (Figure 6.3.6-1) for 1 minute |
| Vibration(1) | 6 dB above acceptance for 3 min in each of 3 axes | 3 dB above acceptance for 2 min in each of 3 axes | Envelope of MPE and minimum spectrum (Figure 8.3.7-1) for 1 min in each of 3 axes |
| Thermal Vacuum | ±10oC beyond acceptance for 8 cycles | ±5oC beyond acceptance for 4 cycles | MPT for 4 cycles |
| Static Load(2) | 1.25 times the limit load for unmanned flight or 1.4 times limit load for manned flight; duration sufficient to record data | 1.25 times the limit load for unmanned flight or 1.25 times limit load for manned flight; duration sufficient to record data | 1.1 times the limit load for bonded, composite, or sandwich structures; duration sufficient to record data |
| Pressure(3) | Not applicable | Not applicable | 1.1 times MEOP for pressurized structures. 1.5 times MEOP for pressurized subsystems including only pressure components. 1.25 times MEOP for pressurized subsystems containing pressure vessels. |
| EMC | 12 dB minimum; duration same as acceptance | 6 dB minimum; duration same as acceptance | 6 dB minimum; 20 min at each space vehicle transmitter frequency for radiated susceptibility |

7.2 Subsystem Development Tests
Vehicles and subsystems are subjected to development tests and evaluations using structural and
thermal development models as may be required to confirm dynamic and thermal environmental cri-
teria for design of subsystems, to verify mechanical interfaces, and to assess functional performance
of deployment mechanisms and thermal control subsystems. Vehicle-level development testing also
provides an opportunity to develop handling and operating procedures as well as to characterize inter-
faces and interactions.
7.2.1 Mechanical Fit Development Tests
For launch and upper-stage vehicles, a mechanical fit, assembly, and operational interface test with
the facilities at the launch or test site is recommended. Flight-weight hardware should be used, if
practical; however, a facsimile or portions thereof may be used to conduct the development tests at an
early point in the schedule in order to reduce the impact of hardware design changes that may be
necessary.
7.2.2 Mode Survey Development Tests
A mode survey test could be conducted at the subsystem level when uncertainty in the analytically
predicted structural dynamic characteristics is judged to be excessive for purposes of structural or
control subsystem design. The test article may be the full vehicle or one or more subsystem seg-
ments, depending on the physical size of the subsystem being tested relative to the physical size of the
test facility and dynamic model verification strategy. Requirements are specified in Reference 17.
7.2.3 Structural Development Tests
Structural tests may be required to verify the stiffness and strength properties and to measure member
loads, stress distributions, deflections, and thermal distortion. Structures with redundant load paths
fall in this category. The stiffness data are of particular interest where significant nonlinear structural
behavior exists. The structural development tests may also be required as an aid to structural
analysis. The member load and stress distribution data may be used to experimentally verify the
structural analysis model. This development test does not replace the structural static load test that is
required for subsystem qualification.
7.2.4 Acoustic Development Tests
Since high-frequency vibration responses are difficult to predict by analytical techniques, acoustic
development testing of the launch, upper stage, and space vehicles subsystems may be necessary to
verify the adequacy of the dynamic design criteria for units. Units that are not installed at the time of
the test shall be dynamically simulated with respect to mass, center of gravity, moments of inertia,
interface stiffness, and geometric characteristics. To demonstrate adequate design margin of the sys-
tem in the launch environment, the subsystem shall be exposed to the qualification or proto-
qualification acoustic levels and durations. To demonstrate adequacy of workmanship, the subsystem
shall be exposed to the acceptance acoustic levels and durations.
7.2.5 Shock Development Tests
Since shock responses are difficult to predict by analytical techniques, shock development testing of
the launch, upper stage, and space vehicles may be necessary to verify the adequacy of the dynamic
design criteria for units. Vehicle units that are not installed at the time of the test shall be dynami-
7 9

cally simulated with respect to mass, center of gravity, moments of inertia, interface stiffness, and
geometric characteristics. All explosive ordnance devices and other mechanisms capable of impart-
ing a significant shock to the vehicle and its mounted assemblies shall be operated to demonstrate
functionality and survivability. Where practical, the shock test shall involve physical separation of
elements being deployed or released. When a significant shock is expected from interfacing subsys-
tems not included on the vehicle under test (such as when a fairing separation causes shock responses
on an upper stage under test), the adaptor subsystem or suitable simulation shall be attached and
appropriate explosive ordnance devices or other means used to simulate the shock imposed. The
pyroshock environment may vary significantly from one ordnance activation to another. Therefore,
the statistical basis given in 3.28 shall be used for estimating maximum predicted environment. Mul-
tiple activations of ordnance devices may be necessary to provide data for better estimates.
7.2.6 Payload Thermal Development Tests
Prior to space vehicle integration, payloads should be subjected to thermal development testing. The
payload thermal vacuum test should include a thermal balance test for thermal model correlation and
demonstration of thermal control hardware. Thermal tests may also be performed to measure the
thermal conductance across important interfaces and the heat loss through critical thermal blankets.
Special tests may also be necessary to verify heat pipe performance in complex configurations or
when heat pipes are in a non-horizontal orientation in system-level thermal testing. Heat pipe opera-
tion shall be verified in subsystem and vehicle ground thermal testing. Heat pipes shall be oriented
such that they operate in ground test orientations. Reflux testing is allowed provided that flight-like
performance is verified at a lower level of assembly. In the case of heat pipes with three-dimensional
bends, a surrogate heat pipe with identical two-dimensional bends may be tested to verify perform-
ance. Heat pipe operation of each three-dimensional flight pipe may be verified in reflux at the unit
level. Because workmanship is not completely demonstrated for three-dimensional heat pipes, every
effort should be made to use two-dimensional heat pipes.
7.3 Test Program for Subsystems
These tests demonstrate that the subsystem will meet its performance and interface requirements. The
baseline is shown in Tables 7.3-1 and 7.3-2. Additional guidance may be found in Reference 32.
7.3.1 Subsystem Performance Test
7.3.1.1 Purpose
The performance test verifies that the mechanical and electrical performance of the subsystem meet
the specification requirements, including compatibility with other subsystems and ground support
equipment, and validates all test techniques and software. Proper operation of all redundant units or
mechanisms must be demonstrated.
7.3.1.2 Mechanical Test
Mechanical devices, valves, deployables, and separation subsystems shall be functionally tested at the
subsystem level in the launch, orbital, or recovery configuration appropriate to the function. Align-
ment checks shall be made where appropriate and feasible. Fit checks shall be made of the subsystem
interfaces using master gages or interface assemblies. The test shall validate that the subsystem per-
forms within maximum and minimum limits under worst-case conditions, including environments,
time, and other applicable requirements. Tests shall demonstrate specified margins of strength,
8 0

torque, and related kinematics and clearances. Where operation in Earth gravity or in an operational
temperature environment cannot be performed, a suitable ground test fixture may be used to permit
operation and performance evaluation. The pass-fail criteria shall be adjusted as appropriate to
account for worst-case maximum and minimum limits that have been modified to adjust for subsys-
tem and ground test conditions. See Reference 7 for further details regarding MMAs.
7.3.1.3 End-to-End Performance Test
These tests shall be performed in accordance with the general requirements stated above. The sub-
system should be in its flight configuration with all units and subsystems connected, except explo-
sive-ordnance elements. The test shall verify the integrity of end-to-end circuits, including functions,
redundancies, deployment circuitry, end-to-end paths, and at least nominal performance, including
radio frequency and other sensor inputs. End-to-end sensor testing may be accomplished with self-
test or coupled inputs.
The test shall be designed to operate all units, primary and redundant, and to exercise all commands
and operational modes to the extent practical. The operation of all thermally controlled units, such as
heaters and thermostats, shall be verified by test. Where control of such units is implemented by sen-
sors, electrical or electronic devices, coded algorithms, or a computer, end-to-end performance testing
shall be conducted. The test shall demonstrate that all commands having precondition requirements
(such as enable, disable, a specific equipment configuration, and a specific command sequence) can-
not be executed unless the preconditions are satisfied. Equipment performance parameters that might
affect end-to-end performance, such as command and data rates, shall be varied over specification
ranges to demonstrate the performance. Autonomous functions shall be verified. Continuous moni-
toring of perceptive parameters, including input and output parameters, and the vehicle main bus by a
power transient-monitoring device and a current monitoring device, shall be provided to detect inter-
mittent failures.
The subsystem shall be operated through a mission profile with all events occurring in actual flight
sequence. This sequence shall include the final countdown, launch, ascent, separation, upper-stage
operation, all appropriate orbital operational modes, and return from orbit, as appropriate. This
sequence shall include live disconnecting of electrical circuits from cable-cutting or connector sepa-
ration. All explosive-ordnance-firing circuits shall be energized and monitored during these events to
verify that the proper energy density is delivered to each device and in the proper sequence. All
measurements that are telemetered shall also be monitored and trended during appropriate portions of
these events to verify proper operations. As a minimum, “a day in the life of the mission test” shall
be run. A portion of this test shall be run with antenna hats removed.
7.3.1.4 Supplementary Requirements
Performance tests shall be conducted before and after the environmental subsystem tests program to
detect equipment anomalies and to assure that performance meets specification requirements. These
tests do not require the mission profile sequence. Sufficient data shall be analyzed to verify the ade-
quacy of the testing and the validity of the data before any change is made to an environmental test
configuration, so that any required retesting can be readily accomplished. During these tests, the
maximum use of telemetry shall be employed for data acquisition, problem identification, and prob-
lem isolation.
8 1

7.3.2 Subsystem Static Load Test
7.3.2.1 Purpose
Static load tests demonstrate the adequacy of structural systems to meet requirements of strength and
stiffness, with the specified test factors, when subjected to simulated critical environments predicted
to occur during their service life (such as temperature, humidity, pressure, and loads).
7.3.2.2 Test Description
Requirements are the same as 6.3.11.2.
7.3.2.3 Test Levels and Duration
Qualification:
a. Level. Unless otherwise specified, the load level for the static load test is 1.4 times limit load
(see 3.21) for manned systems and 1.25 times limit load for unmanned systems.
b. Temperature. Critical flight temperature and load combinations shall be simulated or taken
into account.
c. Duration. The dwell time at each load level shall be sufficient to achieve stable structural
response and record test data such as strain, load, displacement, and temperature.
Protoqualification:
Same as qualification except the load level for static test is 1.25 limit load (3.21) for manned and
unmanned systems.
Acceptance:
A subsystem proof load test shall be conducted for all structural units made of composite materi-
als or having adhesively bonded parts that have not been proof tested at the unit level. The proof
load test is intended to detect material, process, and workmanship defects that could lead to
structural failure. The requirement for the proof load test may be deleted if 6.3.11.3.c is satisfied.
a. Level. Unless otherwise specified, the proof load for flight items shall be1.1 times limit load
(see 3.21).
b. Duration. The dwell time at each load level shall be sufficient to achieve stable structural
response and record test data such as strain, load, displacement, and temperature.
7.3.2.3.1 Test Success Criteria
Requirements are the same as 6.3.11.3.1.
8 2

7.3.2.4 Supplementary Requirements
Other structural test requirements are specified in Reference 6. Static testing is the preferred means of
structural testing. When this is not practical, equivalent tests, such as sine dwell, sine burst, and cen-
trifuge, may be used provided such tests induce similar loading/stress states. Use of alternate tests,
such as sine vibration/sweep, may also be used with approval from the procuring authority provided
these tests generate similar loading/stress state and structural analysis models are correlated.
7.3.3 Subsystem Pressure Test
7.3.3.1 Purpose
A pressurized subsystem is comprised of units that have all been qualified per the above sections.
Qualification testing is not required for lines and fittings that are fabricated using common aerospace
materials and manufacturing processes. With this approach there is no qualification or protoqualifica-
tion required for the assembled pressurized subsystem in regard to pressure design capability. Proof
pressure testing of a pressurized subsystem is required as part of acceptance testing. The proof pres-
sure test detects material and workmanship defects that could result in failure of the pressurized
subsystem.
7.3.3.2 Test Descriptions
Preliminary tests shall be performed, as necessary, to verify compatibility with the test setup and to
ensure proper control of the equipment and test functions. Where pressurized subsystems are
assembled with other than brazed or welded connections, the specified torque values for these
connections shall be verified prior to testing.
In addition to the proof pressure test, the subsystem shall be tested for leakage under propellant
servicing conditions including evacuated internal pressures.
Flow tests at pressure are considered part of performance testing (see 7.3.1).
7.3.3.3 Test Level and Duration
Proof pressure testing is to be followed by leak testing at MEOP per the following requirements:
a. For launch and upper-stage vehicles that contain pressurized structures, the pressurized
subsystem shall be pressurized to a proof pressure that is 1.1 times the maximum expected
operating pressure (MEOP) and held constant for a short dwell time, sufficient to assure that
the proper pressure was achieved within the allowed test tolerance. The test pressure shall
then be reduced to the MEOP for leakage inspection.
b. For space vehicles, each isolated zone of a pressurized subsystem may have an individual
proof pressure level. For zones including pressure vessels, the subsystem zone shall be pres-
surized to a proof pressure, which is 1.25 times the MEOP. For zones without pressure ves-
sels, the proof pressure shall be 1.5 times the MEOP. In each case, the proof pressure shall be
maintained for a time just sufficient to assure that the proper pressure was achieved, and then
the pressure shall be reduced to the MEOP. This sequence shall be followed by inspection
for leakage at the MEOP.
8 3

c. The duration of evacuated propulsion subsystem leakage tests, matching the pressure levels
of propellant servicing conditions, shall not exceed the time that this condition is normally
experienced during propellant loading.
7.3.3.4 Supplementary Requirements
Applicable safety standards shall be followed in conducting all tests. Tests for detecting external
leakage shall be performed at such locations as joints, fittings, plugs, and lines. The acceptable leak-
age rate to meet mission requirements shall be based upon an appropriate analysis. In addition, the
measurement technique shall account for leakage rate variations with pressure and temperature and
have the required threshold, resolution, and accuracy to detect any leakage equal to, or greater than,
the acceptable leak rate. If appropriate, the leakage rate measurement shall be performed at the
MEOP and at operational temperature, with the representative fluid commodity, to account for
dimensional and viscosity changes. Times to achieve thermal and pressure equilibrium, test duration,
and temperature sensitivity shall be determined by an appropriate combination of analysis and devel-
opment test, and the results documented. Leakage detection and measurement procedures may
require vacuum chambers, bagging of the entire vehicle or localized areas, or other special techniques
to achieve the required accuracies. See Reference 18 for further guidance on integration of pressure
components and subsystems.
7.3.4 Subsystem Vibration Test
7.3.4.1 Purpose
Requirements are the same as for vehicles. 8.3.6.1
7.3.4.2 Test Description
Requirements are the same as for vehicles. 8.3.6.2
7.3.4.3 Test Levels and Duration
Requirements are the same as for vehicles. 8.3.6.3
7.3.4.4 Supplementary Requirements
Requirements are the same as for vehicles. 8.3.6.4
In addition, subsystems designed for operation during ascent that are exposed to multiple worst-case
environments such as thermal and vibration are candidates for combined environmental testing.
When such testing is employed, the subsystem shall be tested as close to worst-case flight tempera-
ture as is practical and monitored for temperature performance during vibration exposure.
7.3.5 Subsystem Acoustic Test
7.3.5.1 Purpose
Requirements are the same as for vehicles. 8.3.5.1
7.3.5.2 Test Description
Requirements are the same as for vehicles. 8.3.5.2
8 4

7.3.5.3 Test Levels and Duration
Requirements are the same as for vehicles. 8.3.5.3
7.3.5.4 Supplementary Requirements
Requirements are the same as for vehicles. 8.3.5.4
7.3.6 Subsystem Shock Test
7.3.6.1 Purpose
Requirements are the same as for vehicles. 8.3.4.1
7.3.6.2 Test Description
Requirements are the same as for vehicles. 8.3.4.2
7.3.6.3 Test Activations
Requirements are the same as for vehicles. 8.3.4.3
7.3.6.4 Supplementary Requirements
Requirements are the same as for vehicles. 8.3.4.4
7.3.7 Subsystem Thermal Vacuum Test
Subsystems and functional modules shall be temperature cycled in vacuum with the following
requirements:
a. Those subsystems or portions thereof that cannot be tested as a standalone unit to
performance specifications at the unit level shall be tested at the subsystem level to
unit temperature requirements.
b. Those subsystems or payloads that, at the vehicle level, cannot be tested to their
appropriate thermal environments or cannot meet performance testing requirements
either due to configuration requirements or interaction with other subsystems, shall
be tested at the subsystem level to system temperature requirements.
c. Subsystems that have an external interface shall have their external unit interfaces
tested to performance at temperature extremes in the subsystemt thermal vacuum test
requirements.
7.3.7.1 Purpose
Requirements are the same as for vehicles, 8.3.8.1. For cases where vehicle thermal vacuum testing is
not effective for workmanship screening due to small temperature changes from hot to cold cycles,
then thermal cycle testing at the subsystem level shall be applied. See A.1. When a subsystem
thermal vacuum test is a more effective environment for meeting test objectives specified for the
vehicle thermal vacuum test (such as for payloads and instruments), thermal vacuum testing at the
subsystem level shall be applied. The same shall be applied for thermal balance testing. When
8 5

thermal balance testing is performed at the subsystem level, the approach and requirements of 8.3.7
shall apply.
7.3.7.2 Test Description
Requirements are the same as for vehicles, 8.3.8.2. For cases where the thermal cycle test is applied,
see Section A.1.1.
7.3.7.3 Test Levels and Duration
Requirements are the same as for vehicles, 8.3.8.3, except as noted in 7.3.7. Thermal cycles shall be
added per A.1.1 when system thermal vacuum testing is not effective for workmanship screen.
7.3.7.4 Performance Testing
Requirements are the same as for vehicles, 8.3.8.2. For cases where the thermal cycle test is applied,
see A.1.
7.3.7.5 Supplementary Requirements
Requirements are the same as for vehicles, 8.3.8.2. For cases where the thermal cycle test is applied,
see Section A.1.1.
7.3.8 Subsystem Separation and Deployment Tests
7.3.8.1 Purpose
The separation subsystem test shall be performed as a qualification test to validate the adequacy of
the separation subsystem to meet its performance requirements on such parameters as separation
velocity, acceleration, angular motion, time to clear, clearances, flexible-body distortion and loads,
amount of debris, and shock levels. For a payload fairing, the test also demonstrates the structural
integrity of the fairing and its generic attachments under the separation shock loads environment. The
data from the separation test are also used to validate the analytical method and basic assumptions
used in the separation analysis. The validated method is then used to verify that requirements are met
under worst-case flight conditions.
For deployable subsystems, such as payloads, instruments, etc., qualification and acceptance
deployment tests shall be performed to validate the adequacy of each subsystem to meet its
performance requirements, such as the release function, separation velocity, acceleration, and angular
motion, clearances, flexible-body distortion and loads, amount of debris, and shock levels.
7.3.8.2 Test Description
The test fixtures, including gravity off-loaders, shall duplicate the interfacing structural sections to
simulate the separation subsystem or separating body boundary conditions existing in the flight arti-
cle. The remaining boundary conditions for the separating bodies shall simulate the conditions in
flight, unless the use of other boundary conditions permits an unambiguous demonstration that sub-
system requirements can be met. The test article shall include all attached flight hardware that could
pose a debris threat if detached. When ambient atmospheric pressure may adversely affect the test
results, such as for large fairings, the test shall be conducted in a vacuum chamber, duplicating the
altitude condition encountered in flight at the time of separation. Critical conditions of temperature,
8 6

pressure, or loading due to acceleration shall be simulated or taken into account. For separation
subsystem qualification testing, instrumentation shall include high-speed cameras to record the
motion of specially marked target locations, accelerometers to measure the structural response, and
other environmental data and strain gages to verify load levels in structurally critical attachments.
Critical clearances shall be verified by appropriate measurements. Additional guidelines for moving
mechanical assemblies are provided in Reference 7.
7.3.8.3 Test Activations
Separation and deployment tests shall be conducted to demonstrate that requirements on performance
parameters are met under predicted flight conditions. When critical conditions exist that cannot be
modeled with confidence, additional testing shall be performed to determine the effect of those con-
ditions on the separation or deployment. To validate force or torque margin requirements, tests shall
be conducted to demonstrate that the static and dynamic force margins satisfy the requirements
described in Reference 7. Such a test may occur at a lower level of assembly where the driving and
resisting components are tested for their respective contributions. However, for separating subsys-
tems involving fracture of structural elements, the demonstrated qualification force margin to cause
fracture shall be at least 50%. In addition, debris risk shall be evaluated by conducting a test
encompassing the most severe conditions that can occur in flight.
7.3.8.4 Supplementary Requirements
A post-test inspection for debris shall be conducted on and around the test article.
7.3.9 Subsystem Electromagnetic Compatibility (EMC) Test
7.3.9.1 Purpose
The electromagnetic compatibility test shall demonstrate that the electromagnetic interference char-
acteristics (emission and susceptibility) of the subsystem, under normal operating conditions, do not
result in malfunction of the subsystem. It also demonstrates that the subsystem does not emit, radiate,
or conduct interference, which could result in malfunction of other subsystems.
7.3.9.2 Test Description
The test shall be conducted in accordance with the requirements of Reference 21. All tests shall be
conducted at the payload and bus level on the subsystem external EMC interfaces. Acceptance tests
shall be performed when there is less than a 12 dB qualification margin, or the radiated emissions
requirement is more stringent than 10 dBuV/m, or the subsystem has a passive intermodulation
requirement.
The EMC margin is to be incorporated into the test levels. No additional margin is required if the
Reference 21-1 levels already have the required margin for the interface. Qualification margins of 6
dB are acceptable if the combined test uncertainty, part variation, part degradation at end-of-life, and
workmanship variation is less than 6 dB. Electroexplosive devices and bridge wires have a 20 dB
margin requirement below the DC no-fire value and a 6 dB margin requirement below the RF no-fire
value.
7.3.9.3 Test Levels and Duration
The test levels shall be as follows:
8 7

Qualification: 12 dB
Protoqualification: 6 dB
Acceptance: 6 dB
The test duration shall be 20 minutes at each space vehicle transmitter frequency for radiated
susceptibility. Otherwise, the duration is the greater of three seconds or the unit response time for
susceptibility requirements and 15 milliseconds duration for emission requirements.
7.3.10 Mode Survey Test
A mode survey test shall be conducted to obtain data needed to develop dynamic models for loads
analyses. It may be conducted at the subsystem or at the vehicle level. Reference 17 defines the
scope, application, and test requirements.
7.3.10.1 Purpose
The mode survey test is conducted to experimentally derive a structural dynamic model of a vehicle
or to provide a basis for test-verification of an analytical model. After upgrading analytically to the
flight configuration (such as different propellant loading and differences between flight and test unit
mass properties), this model is used in analytical simulations of flight loading events. The computed
loads are used to determine structural margins and adequacy of the structural static test loading con-
ditions (7.3.2). They are, therefore, critical for verification of vehicle structural integrity and qualifi-
cation of the structural subsystem as flight-ready. Where practical, a mode survey test is also per-
formed to define or verify models used in the final preflight evaluation of structural dynamic effects
on control subsystem precision, stability, and pointing performance.
7.3.10.2 Test Description
The test article shall consist of flight-quality structure with assembled units, payloads, and other
major subsystems, and shall contain actual or simulated liquids. For large vehicles, complexity and
testing practicability may dictate that tests be performed at the subsystem level. Mass simulators may
be used to represent flight items when their attachment-fixed resonances have been demonstrated by
test to occur above the frequency range of the mode survey test. Dynamic simulators may be used
for items that have resonances within the frequency range of interest if they are accurate dynamic
representations of the flight item. Alternatively, mass simulators may be used if flight-quality items
are subjected separately to mode survey testing that meet test requirements. All mass simulators are
to include realistic simulation of interface attach structure, and artificial stiffening of the test structure
shall be avoided.
The data obtained in the modal survey shall be adequate to define the mode shapes, natural frequen-
cies, and damping values for all modes that occur in the frequency range of interest, typically up to 70
Hz. In addition, the first two modes in each lateral coordinate plane, the first axial mode and the first
torsional mode, shall be acquired even if their frequencies lie outside the specified test range. The
quality of the measured modes must be judged by computing the mass-weighted orthogonality of the
8 8

mode shapes. As a goal, the off-diagonal terms of the unit normalized generalized mass matrix
should be equal to or less than 0.10.
7.3.10.3 Test Levels
The test is generally conducted at response levels that are low compared to the expected flight levels.
Limited testing shall be conducted to evaluate nonlinear behavior, with a minimum of three levels
used when significant nonlinearity is identified.
7.3.10.4 Supplementary Requirements
Because of their criticality to achieving a successful test, appropriate pretest analyses and
experimentation shall be performed to:
a. Establish test instrumentation requirements.
b. Evaluate the test stand and fixturing to preclude any boundary condition uncertainties
that could compromise test objectives.
c. Verify that mass simulators have no resonances within the frequency range of
interest.
d. Establish flexibility to add or modify instrumentation during test to account for unex-
pected modes.
8 9

THIS PAGE INTENTIONALLY LEFT BLANK

8. Vehicle Test Requirements
8.1 General Requirements
The vehicle-level test baseline assumes that all unit-level and subsystem tests have been performed in
accordance with this document. Tests that are conducted as acceptance tests for vehicle elements
(such as alignments, instrument calibrations, antenna patterns, and mass properties) shall also have
been conducted. All flight equipment and software shall be installed prior to beginning vehicle-level
testing.
The sequence of tests performed at the vehicle level is shown in Table 8.3-1 for qualification and
protoqualification vehicles and in Table 8.3-2 for acceptance vehicles. An overview of vehicle test
level margins and durations described in this section is shown in Table 8.3-3.
System-level tests shall be the same as subsystem tests for end-to-end performance, configuration
change, command and telemetry performance, and external interfaces to the ground and LV compo-
nent, except that the actual compatibility interface testing to the other external system components
need not be done at temperature if testing against a simulator at temperature was accomplished at a
lower level of assembly.
Table 8.3-1. Vehicle Qualification and Protoqualification Test Summary
Suggested Launch Upper-stage Space
Test Section Sequence Vehicle Vehicle Vehicle
Inspection 4.6 1, 13 R R R
Performance (1)(4) 8.3.1 2, 12 R R R
Pressure and Leak (5) 8.3.2 3, 7, 10 R R R
EMC(2) 8.3.3 4 or 11 (2) R R R
Shock(7) 8.3.4 6 R R R
Acoustic 8.3.5
or 5 ER R R
Random Vibration (3) 8.3.6
Thermal Balance 8.3.7 8 -- ER R
Thermal Vacuum 8.3.8 9 -- ER R
Mode Survey (6) 8.3.9 Any ER ER R
R Required
ER Evaluation required (see 6.3)
(1) Performance tests conducted prior to, during, and following each environmental test
as appropriate.
(2) EMC testing, sequence 11, shall be conducted when there are radiated emission
requirements below 10 dBuV/m or there is a requirement on passive intermodulation
levels.
(3) In some cases, vibration may be used in place of acoustics for vehicle weights under
400 lbs. (180 Kg).
(4) Deployments and critical clearance shall be verified (see 7.3.8).
(5) Requirement is met by subsystem-level testing unless a modification or repair has
occurred.
(6) Mode Survey may be performed at the subsystem level if the test is not feasible at
the system level. See Table 7.3-1.
(7) Preferred at the vehicle level; if not feasible, performed at the subsystem level.
9 1

| Test | Section | Suggested Sequence | Launch Vehicle | Upper-stage Vehicle | Space Vehicle |
| --- | --- | --- | --- | --- | --- |
| Inspection | 4.6 | 1, 13 | R | R | R |
| Performance (1)(4) | 8.3.1 | 2, 12 | R | R | R |
| Pressure and Leak (5) | 8.3.2 | 3, 7, 10 | R | R | R |
| EMC(2) | 8.3.3 | 4 or 11 (2) | R | R | R |
| Shock(7) | 8.3.4 | 6 | R | R | R |
| Acoustic or Random Vibration (3) | 8.3.5 8.3.6 | 5 | ER | R | R |
| Thermal Balance | 8.3.7 | 8 | -- | ER | R |
| Thermal Vacuum | 8.3.8 | 9 | -- | ER | R |
| Mode Survey (6) | 8.3.9 | Any | ER | ER | R |

Table 8.3-2 Vehicle Acceptance Test Summary
Reference Suggested Launch Upper Stage Space
Test Paragraph Sequence Vehicle Vehicle Vehicle
Inspection 4.6 1, 12 R R R
Performance (1) (4) 8.3.1 2, 11 R R R
Pressure and Leak (5) 8.3.2 3, 7, 9 R R ER
EMC(2) 8.3.3 4 or 10 (2) ER ER ER
Shock 8.3.4 6 ER ER R
Acoustic 8.3.5
or 5 ER R R
Vibration (3) 8.3.6
Thermal Vacuum 8.3.8 8 -- ER R
R Required
ER Evaluation required (see 6.3)
(1) Performance tests shall be conducted prior to, during, and following each
environmental test as appropriate.
(2) EMC testing (sequence 4 or 10) required when the qualification margin is not
met, or there are radiated emission requirements below 10 dBuV/m, or there is a
requirement on passive intermodulation levels.
(3) In some cases, vibration may be used in lieu of acoustics for vehicles under 400
lbs (180 Kg).
(4) Deployments and critical clearance shall be verified (see 7.3.8).
(5) Requirement is met by subsystem level testing unless a modification or repair
has occurred
Table 8.3-3 Vehicle Test Level Margins and Duration
Test Qualification Protoqualification Acceptance
Shock 1 activation of all shock-producing 1 activation of all shock- 1 activation of significant shock-
events; 2 additional activations of producing events; 1 additional producing events
significant events activation of significant events
Acoustic(1) 6 dB above acceptance for 3 min 3 dB above acceptance for 2 min Envelope of MPE and minimum
spectrum
(Figure 6.3.6-1) for 1 minute
Vibration(1) 6 dB above acceptance for 3 dB above acceptance for 2 min Envelope of MPE and minimum
3 min in each of 3 axes in each of 3 axes spectrum (Figure 8.3.7-1) for 1
min in each of 3 axes
Thermal ±10oC beyond acceptance for 8 ±5oC beyond acceptance for MPT for 4 cycles
Vacuum(2) cycles 4 cycles
Pressure(3) Not applicable Not applicable Proof pressure as specified in
7.3.3.3 for pressurized
subsystems(3).
Leak tests at MEOP per 7.3.3.3
EMC 12 dB minimum duration same as 6 dB minimum duration same as 6 dB minimum 20 minutes at
acceptance acceptance each space vehicle transmitter
frequency for radiated
susceptibility
(1) See B.1.3 for vehicles with effective duration greater than 15 seconds.
(2) See A.1.1 if vehicle thermal cycle testing is performed.
(3) Requirement met by subsystem level testing unless a modification or repair has occurred
8.2 Vehicle Development Tests
Vehicles are subjected to development tests and evaluations using structural and thermal development
models as may be required to confirm dynamic and thermal environmental criteria for design of sub-
systems, to verify mechanical interfaces, and to assess functional performance of deployment mecha-
nisms and thermal control subsystems. Vehicle-level development testing also provides an opportu-
9 2

| Test | Reference Paragraph | Suggested Sequence | Launch Vehicle | Upper Stage Vehicle | Space Vehicle |
| --- | --- | --- | --- | --- | --- |
| Inspection | 4.6 | 1, 12 | R | R | R |
| Performance (1) (4) | 8.3.1 | 2, 11 | R | R | R |
| Pressure and Leak (5) | 8.3.2 | 3, 7, 9 | R | R | ER |
| EMC(2) | 8.3.3 | 4 or 10 (2) | ER | ER | ER |
| Shock | 8.3.4 | 6 | ER | ER | R |
| Acoustic or Vibration (3) | 8.3.5 8.3.6 | 5 | ER | R | R |
| Thermal Vacuum | 8.3.8 | 8 | -- | ER | R |

| Test | Qualification | Protoqualification | Acceptance |
| --- | --- | --- | --- |
| Shock | 1 activation of all shock-producing events; 2 additional activations of significant events | 1 activation of all shock- producing events; 1 additional activation of significant events | 1 activation of significant shock- producing events |
| Acoustic(1) | 6 dB above acceptance for 3 min | 3 dB above acceptance for 2 min | Envelope of MPE and minimum spectrum (Figure 6.3.6-1) for 1 minute |
| Vibration(1) | 6 dB above acceptance for 3 min in each of 3 axes | 3 dB above acceptance for 2 min in each of 3 axes | Envelope of MPE and minimum spectrum (Figure 8.3.7-1) for 1 min in each of 3 axes |
| Thermal Vacuum(2) | ±10oC beyond acceptance for 8 cycles | ±5oC beyond acceptance for 4 cycles | MPT for 4 cycles |
| Pressure(3) | Not applicable | Not applicable | Proof pressure as specified in 7.3.3.3 for pressurized subsystems(3). Leak tests at MEOP per 7.3.3.3 |
| EMC | 12 dB minimum duration same as acceptance | 6 dB minimum duration same as acceptance | 6 dB minimum 20 minutes at each space vehicle transmitter frequency for radiated susceptibility |

nity to develop handling and operating procedures as well as to characterize interfaces and
interactions.
8.2.1 Mechanical Fit Development Tests
For launch and upper-stage vehicles, a mechanical fit, assembly, and operational interface test with
the facilities at the launch or test site is recommended. Flight-weight hardware should be used if
practical; however, a facsimile or portions thereof may be used to conduct the development tests at an
early point in the schedule in order to reduce the impact of hardware design changes that may be
necessary.
8.2.2 Mode Survey Development Tests
A development mode survey test should be conducted at the vehicle level when uncertainty in
analytically predicted structural dynamic characteristics is judged to be excessive for purposes of
structural or control subsystem design, and an early identification of problem areas is desired. The
test article may be the full vehicle or one or more substructures depending on size and complexity.
Such a development test does not replace the mode survey test(s) required for the Verification Load
Cycle (Reference 17), and the requirements as summarized in 8.3.9.
8.2.3 Structural Development Tests
Structural tests may be required to verify the stiffness and strength properties and to measure member
loads, stress distributions, deflections, and thermal distortion. Typical examples are structures with
redundant load paths or new technology implementation. This development test does not replace the
structural static load test that is required for subsystem qualification.
8.2.4 Acoustic Development Tests
Since high-frequency vibration responses are difficult to predict by analytical techniques, acoustic
development testing of the launch, upper-stage, and space vehicles may be necessary to verify the
adequacy of the dynamic design criteria for units. Vehicle units that are not installed at the time of
the test should be dynamically simulated with respect to mass, center of gravity, moments of inertia,
interface stiffness, and geometric characteristics. The test article should be exposed to the maximum
predicted flight levels and instrumented at unit and other points of interest to obtain vibration
responses for use in verifying unit predictions. Supporting structures such as payload or upper-stage
adapters should be included, and responses evaluated to understand structurally borne energy that
contributes to the response at nearby units.
8.2.5 Shock Development Tests
Since high-frequency shock responses are difficult to predict by analytical techniques, shock devel-
opment testing of the launch, upper stage, and space vehicles may be necessary to verify the adequacy
of the dynamic design criteria for units. Vehicle units that are not installed at the time of the test
should be dynamically simulated with respect to mass properties, interface stiffness, and geometry.
All explosive ordnance devices and other mechanisms capable of imparting a significant shock to the
vehicle and its mounted assemblies should be operated to demonstrate functionality and survivability.
Where practical, the shock test should involve physical separation of elements being deployed or
released. When a significant shock is expected from interfacing subsystems not included on the vehi-
cle under test (such as when a fairing separation causes shock responses on an upper stage under test),
9 3

the adaptor subsystem or suitable simulation shall be attached and appropriate explosive ordnance
devices or other means used to simulate the shock imposed. The pyroshock environment may vary
significantly between ordnance activations. Therefore, the statistical basis given in 3.28 shall be used
for estimating maximum expected and extreme spectra. Multiple activations of ordnance devices
may be necessary to provide data for improved estimates.
8.2.6 Thermal Balance Development Tests
A thermal balance development test is performed to verify the analytical thermal modeling of launch,
upper-stage, or space vehicles, and demonstrate the ability of the thermal control subsystem to main-
tain temperature limits. For vehicles in which thermally induced structural distortions are critical to
mission success, the thermal balance test also evaluates alignment concerns. The test vehicle should
consist of flight hardware or a thermally equivalent structure with addition of equipment panels,
thermal control insulation, finishes, and thermally equivalent models of electrical, electronic, pneu-
matic, and mechanical units. Testing should be conducted in a space simulation test chamber capable
of simulating the ascent, transfer orbit, and orbital thermal vacuum conditions as may be appropriate.
The test consists of simulating different environmental and operational modes and collecting steady-
state and transient thermal data to correlate the thermal analytic model and verify performance of the
thermal control hardware.
8.2.7 Transportation and Handling Development Tests
The handling and transport of launch, upper-stage, space vehicles, or their sub-tier elements is nor-
mally conducted to result in dynamic environments well below those expected for launch and flight.
However, since these environments are difficult to predict, it is often necessary to conduct a devel-
opment test of potentially significant handling and transportation configurations to determine
worst-case dynamic inputs. Such a test should use a development model of the item or a simulator
that has at least the proper mass properties and RFI shielding effectiveness, instrumented to measure
responses of the item. In particular, a drop test representative of a maximum credible operational
occurrence should be conducted to demonstrate protection of the item in the handling apparatus and
validate design of the shipping container. The data should be sufficient to determine whether the
environments are benign relative to the design requirements, or to provide a basis for an analysis to
demonstrate lack of damage, or to augment qualification and acceptance testing, if necessary.
8.2.8 Wind Tunnel Development Tests
Flight vehicle aerodynamic and aero-thermal data are needed to establish that the vehicles survive
flight, and function properly under the imposed loads. For flight vehicles with a new or significantly
changed aerodynamic design, the following wind tunnel tests shall be conducted:
a. Force and Moment Tests. These tests provide the resultant aerodynamic forces and
moments acting on the vehicle during the high-dynamic-pressure region of flight. Data from
these tests are used in both structural and control subsystem design and in trajectory analysis.
b. Steady-State Pressure Tests. These tests determine the spatial distribution of the
steady-state component of the pressures imposed on the vehicle’s external surfaces during the
high-dynamic-pressure region of flight. These data are used to obtain the axial air load dis-
tributions, which are used to evaluate the static-elastic characteristics of the vehicle. These
data along with fluctuating pressures of the external skin environment are also used in com-
9 4

partment venting analyses to determine burst and collapse pressures imposed on the vehicle
structure. The design and testing of the payload fairing structure are particularly dependent
upon high-quality definition of these pressures.
c. Aerodynamic Heating Tests. These tests determine the heating effects due to fin and fuse-
lage junctures, drag (friction), angle of attack, flow transition, shock wave impingement,
proximity effects for multibody vehicles, and surface discontinuities.
d. Base Heating Tests. These tests determine the heating effects due to thermal radiation, mul-
tiplume recirculation convection, plume-induced flow separation on the vehicle body, and the
base flow field.
e. Thruster Plume-Impingement Heating Tests. These tests determine the heating effects
and contamination due to impingement of the thruster plumes.
f. Transonic and Supersonic Buffet and Aerodynamic Noise Tests. These tests define the
spatial distribution of the unsteady or fluctuating component of the pressures imposed on the
vehicle external surfaces during the high-dynamic-pressure region of flight. These data are
used to obtain the dynamic airloads acting to excite the various structural modes of the vehi-
cle and are used in aeroelastic, flutter, and vibroacoustic analyses. These data are also used in
compartment venting analyses to determine burst and collapse pressures imposed on the
vehicle structure.
g. Ground-Wind-Induced Oscillation Tests. These tests define the resultant forces and
moments acting on the vehicle prior to launch when it is exposed to the ground-wind envi-
ronment. Flexible models or elastically mounted rigid models are used to simulate at least
the first cantilever-bending mode of the vehicle. Nearby structures or terrain, which may
influence the flow around the vehicle, shall also be simulated.
h. Aerodynamic Staging Tests. These tests determine the forces and moments acting on the
core vehicle and solid rocket motors (SRM) that are oriented in a series of representative
positions encountered during SRM booster separation. Data from these tests are used in stage
performance analysis.
8.3 Test Program for Flight Vehicles
Testing at the system level is composed of performance tests that are performed under ambient
conditions and during or after environmental exposure.
The vehicle configuration shall contain all of its flight subsystems with flight software, and shall
interface with external hardware and facilities and or simulators for external interface verification.
Additional guidance may be found in Reference 32.
9 5

8.3.1 Vehicle Performance and Functional Tests
8.3.1.1 Purpose
Performance and functional testing verify that the mechanical and electrical operations of the vehicle
meet requirements, including interoperability with ground support equipment, and ground station
assets.
8.3.1.2 Mechanical Test
Mechanical devices, valves, deployables, and separation subsystems shall be functionally tested at the
vehicle level in the launch, orbital, or recovery configuration appropriate to the function. Alignment
checks shall be made where appropriate and feasible. Fit checks shall be made of the vehicle physical
interfaces. Testing shall validate that the vehicle performs within maximum and minimum limits
under worst-case conditions, including environments, time, and other applicable requirements. Tests
shall demonstrate positive margins of strength, force/torque, and related kinematics and clearances.
Where operation in earth gravity or in an operational temperature environment cannot be performed, a
suitable ground test fixture may be used to permit operation and performance evaluation. The pass-
fail criteria shall be adjusted, as appropriate, to account for worst-case maximum and minimum limits
that have been modified to adjust for ground test conditions. For additional details concerning
deployment testing, see 7.3.8 and Reference 7.
8.3.1.3 End-to-End Performance Test
Vehicle performance testing is conducted to verify that the vehicle satisfies performance and func-
tional requirements before and after exposure to environmental testing, hardware and software
changes, hardware removal, and replacements. The vehicle shall be in its flight configuration with all
units and subsystems connected. The test shall verify the integrity of end-to-end circuits, including
functions, redundancies, deployment circuitry, end-to-end paths, and performance, including radio
frequency and other sensor inputs. End-to-end sensor testing may be accomplished with self-test or
coupled inputs.
The test shall be designed to operate all units, primary and redundant, and to exercise all commands
and operational modes to the maximum extent practical. The operation of all thermally controlled
units, such as heaters and thermostats, shall be verified by test. Where control of such units is imple-
mented by sensors, electrical or electronic devices, coded algorithms, or a computer, end-to-end per-
formance testing shall be conducted. The test shall demonstrate that all commands having precondi-
tion requirements (such as enable, disable, a specific equipment configuration, and a specific com-
mand sequence) cannot be executed unless the preconditions are satisfied. Equipment performance
parameters that might affect end-to-end performance, such as command and data rates, shall be varied
over specification ranges to demonstrate the performance. Autonomous functions shall be verified.
Continuous monitoring of perceptive parameters, including input and output parameters, and the
vehicle main bus by a power transient and a current monitoring device, shall be provided to detect
intermittent failures.
The vehicle shall be operated through a mission profile with all events occurring in actual flight
sequence. This sequence shall include the final countdown, launch, ascent, separation, upper-stage
operation, all appropriate orbital operational modes, and return from orbit as appropriate. All explo-
sive ordnance firing circuits shall be energized and monitored during these events to verify that the
9 6

proper energy density is delivered to each device and in the proper sequence. Telemetry shall be
monitored and trended during appropriate portions of these events to verify proper operations. Suffi-
cient data shall be analyzed to verify the adequacy of the testing and the validity of the data before
any change is made to an environmental test configuration so that any required retesting can be read-
ily accomplished. The final software testing shall demonstrate that the hardware and software con-
figuration can meet the defined functional and performance specifications in a worst-case stressing
environment.
During one test in the acceptance flow, the vehicle shall be subjected to a “plugs-out” test. The pur-
pose of this test is to demonstrate that all vehicle systems are in an “as near-flight configuration” as is
practical. At the start of the test, umbilicals are disconnected and the vehicle is on flight batteries or
flight-like batteries. The vehicle shall be operated through a basic set of functional checkouts and
telemetry responses.
8.3.1.4 Supplementary Requirements
An end-to-end mission readiness test should be performed with the space vehicle in the factory
operating under ground station control. This test should demonstrate that the combined space/ground
system can perform the mission before flight, and to detect anomalies unique to this architecture. This
end-to-end test should be run on the mission timeline and in the mission sequence. This test should
be associated with the final integrated system test so that final flight hardware and flight/ground
software loads are tested together. The test should also include mission tasking and data
dissemination. The synergistic effects of combined environmental stresses and stressing mission
scenarios should be considered.
8.3.2 Vehicle Pressure and Leakage Test
8.3.2.1 Purpose
These tests demonstrate the capability of pressurized subsystems to meet the specified pressure and
leakage rate requirements. Vehicle-level testing for proof pressure need only include those zones or
portions of the pressure system that are not fully assembled until the vehicle is complete or the
integrity of which is suspect after the completion of the subsystem level testing. Leak testing at
MEOP in pressurized subsystems shall be conducted.
Flow tests at pressure are considered part of performance testing (see 8.3.1).
8.3.2.2 Test Description
See 7.3.3.2
8.3.2.3 Test Levels and Durations
Pressurized subsystems shall have a capability to repeat testing of 7.3.3.3 at the vehicle level in the
event of a modification or repair to the subsystem.
8.3.2.4 Supplementary Requirements
Applicable safety standards shall be followed in conducting all tests. Tests for detecting external
leakage shall be performed at such locations as joints, fittings, plugs, and lines. The acceptable leak-
9 7

age rate to meet mission requirements shall be based upon an appropriate analysis. In addition, the
measurement technique shall account for leakage rate variations with pressure and temperature and
have the required threshold, resolution, and accuracy to detect any leakage equal to or greater than the
acceptable leak rate. If appropriate, the leakage rate measurement shall be performed at the MEOP
and at operational temperature, with the representative fluid commodity, to account for dimensional
and viscosity changes. Times to achieve thermal and pressure equilibrium, test duration, and temper-
ature sensitivity shall be determined by an appropriate combination of analysis and development test,
and the results documented. Leakage detection and measurement procedures may require vacuum
chambers, bagging of the entire vehicle or localized areas, or other special techniques to achieve the
required accuracies. See Reference 4 for further guidance on integration of pressure components and
subsystems.
8.3.3 Vehicle Electromagnetic Compatibility Test
8.3.3.1 Purpose
The electromagnetic compatibility test demonstrates electromagnetic compatibility of the vehicle.
EMC testing at the vehicle level assumes that full EMC testing in accordance with Reference 21 has
been accomplished at the unit and/or subsystem level and that bus and payload EMC testing has also
occurred in accordance with the tests described herein.
8.3.3.2 Test Description
The operation of the vehicle and selection of instrumentation shall be suitable for determining the
margin against malfunctions and unacceptable or undesired responses due to electromagnetic
incompatibilities.
The test shall demonstrate satisfactory electrical and electronic equipment operation in conjunction
with the expected electromagnetic radiation from other subsystems or equipment, such as from other
vehicle elements and ground support equipment. The vehicle shall be subjected to the required tests
while in the launch, orbital, and return-from-orbit configurations, and in all possible operational
modes, as applicable. Special attention shall be given to areas indicated to be marginal by unit-level
test and analysis. Potential electromagnetic interference between the test vehicle and other subsys-
tems shall be measured. The tests shall be conducted according to the requirements of Reference 21.
The tests shall include, but not be limited to, nine main segments:
a. Radio frequency (RF) self-compatibility (all receivers and transmitters receiving and
transmitting through flight antennas without antenna hats)
b. Power quality
c. Radiated emissions
d. Radiated susceptibility
e. Conducted emissions
f. Power transients
9 8

g. Magnetic moments
h. Critical circuit margins
i. Umbilical separation test
Explosive-ordnance devices having bridge wires, but otherwise inert, shall be installed in the vehicle
and monitored during all tests.
Acceptance tests shall be performed when the qualification margin is not met, or the radiated emis-
sions requirement is more stringent than 10 dBuV/m, or the system has a passive intermodulation
requirement.
The EMC margin is to be incorporated in to the test levels. No additional margin is required if the
Reference 21 levels already have the required margin for the interface. Qualification margins of 6 dB
are acceptable if the combined test uncertainty, part variation, part degradation at end-of-life, and
workmanship variation is less than 6 dB. Electroexplosive devices and bridge wires have a 20 dB
margin requirement below the DC no-fire value and a 6 dB margin requirement below the RF no-fire
value.
8.3.3.3 Test Levels and Duration
The test levels shall be as follows:
Qualification: 12 dB
Protoqualification: 6 dB
Acceptance: 6 dB
The test duration shall be 20 minutes at each space vehicle transmitter frequency for radiated
susceptibility. Otherwise, the duration is the greater of three seconds or the unit response time for
susceptibility requirements and 15 milliseconds duration for emission requirements.
8.3.3.4 Supplementary Requirements
For guidance on testing rationale for satellite hardness and survivability refer to Reference 29.
8.3.3.4.1 Vehicle Qualification and Protoqualification
EMC testing Sequence 11 in the list of requirements shown in Table 8.3-1 shall be conducted when
there are radiated emission requirements below 10 dBuV/m or there is a requirement on passive
intermodulation levels. Sequence 4 is the preferred sequence as this minimizes the risk of repeating
the mechanical environmental testing in the event of a failure and subsequent rework. If Sequence 11
is performed, conducting Sequence 4, in addition to Sequence 11, should be considered to reduce the
risk of repeating the mechanical environmental testing in the event of a failure and subsequent
rework.
9 9

8.3.3.4.2 Vehicle Acceptance
EMC acceptance testing Sequence 4 or 10 in the list of requirements shown in Table 8.3-2 shall be
conducted when the qualification margin is not met, or there are radiated emission requirements
below 10 dBµV/m, or there is a requirement on passive intermodulation levels. EMC testing
Sequence 10 shall be conducted when there are radiated emission requirements below 10 dBµV/m or
there is a requirement on passive intermodulation levels. Sequence 4 is the preferred sequence as this
minimizes the risk of repeating the mechanical environmental testing in the event of a failure and
subsequent rework. If Sequence 10 is performed, conducting Sequence 4, in addition to Sequence 10,
should be considered to reduce the risk of repeating the mechanical environmental testing in the event
of a failure and subsequent rework.
8.3.4 Vehicle Shock Test
8.3.4.1 Purpose
Shock testing demonstrates the capability of the vehicle to withstand or, if appropriate, to operate
through the induced shock environments. Shock testing also yields data to validate the maximum
expected unit shock requirements.
8.3.4.2 Test Description
The vehicle shall be supported and configured to allow flight-like dynamic response of the vehicle
with respect to amplitude, frequency content, and paths of transmission. Support of the vehicle may
vary during the course of a series of shock tests in order to reflect the configuration at the time of
each shock event. Test setups shall avoid undue influence of test fixtures, and prevent re-contact of
separated items.
In the shock test, or series of shock tests, the vehicle shall be subjected to shock transients that cause
the extreme expected shock environments to the extent practical. Shock events to be considered
include separations and deployments initiated by explosive ordnance or other devices, as well as
impacts and suddenly applied or released loads that may be significant for unit dynamic response
(such as due to an engine transient, parachute deployment, and vehicle landing). All devices on the
vehicle capable of imparting significant shock to the vehicle shall be activated. Those potentially
significant shock sources not on the vehicle under test, such as on an adjoining payload fairing or a
nearby staging joint, shall also be actuated or simulated and applied through appropriate interfacing
structures. Dynamic instrumentation shall be installed to measure shock responses in three orthogo-
nal directions at attachments of selected units.
8.3.4.3 Test Activations
Qualification: All explosive-ordnance devices and other potentially significant
shock-producing devices or events, including those from sources not
installed on the vehicle under test, shall be activated at least one
time, or simulated, as appropriate. The significant shock events shall
be activated two additional times to provide for variability in the
vehicle test and to provide data for prediction of maximum and
extreme expected shock environments for units. Activation of both
primary and redundant devices shall be carried out in the same
sequence as they are intended to operate in service.
10 0

Protoqualification: Same as qualification except only one additional activation of sig-
nificant shock producing events is required.
Acceptance: One activation of significant shock-producing events is required.
8.3.4.4 Supplementary Requirements
Electrical and electronic units shall be operating and monitored to the maximum extent practical and
safe. Continuous monitoring of several perceptive parameters, including input and output parameters,
and the vehicle main bus by a power transient-monitoring device, shall be provided to monitor power
quality and detect intermittent failures.
8.3.5 Vehicle Acoustic Test
8.3.5.1 Purpose
The acoustic test demonstrates the ability of the vehicle to endure acoustic acceptance testing and
meet requirements during and after exposure to the applicable acoustic environment of flight. Except
for items whose environment is dominated by structure-borne vibration, the acoustic test also verifies
the adequacy of unit vibration qualification levels, and serves as a qualification test and envi-
ronmental stress screen for items not tested at a lower level of assembly.
8.3.5.2 Test Description
The vehicle in its ascent configuration shall be installed in an acoustic test facility capable of gener-
ating sound fields or fluctuating surface pressures that induce vehicle vibration environments suffi-
cient for vehicle qualification. The vehicle shall be mounted on a flight-type support structure. Sig-
nificant fluid and pressure conditions shall be replicated to the extent practical. Appropriate dynamic
instrumentation shall be installed to measure vibration responses at attachment points of critical and
representative units. Control microphones shall be placed at a minimum of four well-separated loca-
tions, preferably at one-half the distance from the test article to the nearest chamber wall, but no
closer than 0.5 m (20 in.) to both the test article surface and the chamber wall. When test article size
exceeds facility capability, the vehicle may be appropriately subdivided and acoustically tested as one
or more subsystems or assemblies.
8.3.5.3 Test Level and Duration
The basic test levels and duration required for vehicles exposed to the liftoff and ascent acoustic
excitation, effective duration of 15 seconds (see 3.12), are as follows:
Qualification: 6 dB above acceptance for 3 min
Protoqualification: 3 dB above acceptance for 2 min
Acceptance: Envelope of the maximum predicted environment and minimum
workmanship level shown in Figure 6.3.6-1 for 1 min
8.3.5.4 Supplementary Requirements
During the test, all electrical and electronic units used during launch, ascent, and on orbit, or that may
be especially susceptible to vibroacoustic failure modes, shall be electrically energized and sequenced
10 1

through operational modes to the maximum extent practical, with the exception of units that may
sustain damage if energized. Continuous monitoring of appropriate perceptive parameters, including
input and output parameters, and the vehicle main bus by a power transient-monitoring device and a
current monitoring device, shall be provided to detect intermittent failures.
See B.1.5 for discussion of a damage-based approach to the analysis of flight acoustic data for deter-
mining the adequacy of established acceptance and qualification testing when new flight data bring
the adequacy of the MPE spectrum into question.
8.3.5.5 Options for Acoustic Testing
8.3.5.5.1 Flightproof Acoustic Test
For program procurements of one or two vehicles, flightproof acoustic tests may be exercised as an
option. This approach subjects the vehicle to a flightproof test that is an enhanced acceptance test
using protoqualification levels (+3 dB), over the acceptance duration (one minute). Confidence is
gained that each flightproof vehicle meets performance requirements in flight after having success-
fully passed testing beyond the maximum expected flight environments (MPE). Thus, flightproof
testing is a check on the adequacy of each flight item, considering build variability or defects
introduced due to handling or testing. In addition, flightproof reduces potential fatigue failures for
retest due to the reduced exposure time. Since flightproof testing does not validate an acceptance test
program (see B.1.2), acoustic testing of subsequent vehicle is performed, again, at flightproof levels.
8.3.5.5.2 Option for Deletion of Acceptance Acoustic Test
For satellite programs in excess of five vehicles with a consistent block design, an option to delete the
acceptance acoustic test may be exercised based on the level of accepted residual risk. The contractor
shall assess risk and provide documentation of the accepted risk of deleting the acoustic test. The
contractor shall provide a report documenting the design and manufacture pedigrees and vehicle test
results from the five previous vehicles in block to support the justification for execution of this
option. Reference 31 provides information necessary to assess deleting the vehicle acoustic test after
five sequential anomaly-free acoustic tests under the following conditions:
1. Consistent dynamic response to acoustics with no Category 1 or 2 anomalies noted and
associated with the five preceding and sequential acoustic tests, including state changes
identified in post-test functional/performance tests. Category 1 and 2 anomalies are
discussed in B.3.
2. No flight anomalies potentially associated with dynamic environments during the first
180 days flight experience.
3. Consistent block design with no primary or secondary structural or harnessing design
changes within block.
4. Demonstrated production stability on all vehicles within block. Detailed production
stability criteria are discussed in Reference 31.
10 2

5. The contractor shall provide a risk assessment for customer approval prior to executing
this option.
Category 1 and 2 anomalies are discussed in B.3
Once these criteria are satisfied and a decision to delete the acoustic test is implemented, it is neces-
sary to continue tracking program stability metrics to ensure the no-test status remains valid across
builds. The test should be reinstated when these criteria are violated.
8.3.5.5.3 Alternative Power-On Acoustic Test Configuration
A baseline test requires the vehicle be electrically energized to the maximum extent feasible while
maintaining vehicle safety. When this is not feasible or safe, the test may be run in other electrical
configurations. In this case the supplier shall provide technical rationale and a risk assessment to the
customer for approval. The risk assessment should consider latent defects that require synergistic
environmental stress and electrical power for detection, which was not provided during the test.
8.3.5.5.4 Option to Perform Acoustic Test After Thermal Vacuum
As indicated in Tables 8.3-1 and 8.3-2, the suggested sequence shows the acoustic test preceding the
thermal vacuum test. On acceptance test programs where schedule and hardware availability suggest
a change in the test sequence, the acoustic test may be performed following the thermal vacuum test.
This represents an increased risk to the program as the thermal vacuum test, the most perceptive envi-
ronmental test, is not the last environmental test performed. The risk is due to the possibility of
defects escaping that would not be detected without a subsequent thermal vacuum test. The rese-
quencing may involve moving the associated shock test. If the modified test sequence is adopted, the
acoustic should be run after shock and before post-test functional tests. The contractor shall conduct
and provide a technical justification and a risk assessment to the customer for approval. The risk
assessment should consider latent defects that require a subsequent thermal test for detection, which is
no longer available.
A major purpose of the protoqualification, or flightproof, testing is design verification requiring
maximum test effectiveness. As a result, the modified sequence approach is not available for pro-
toqualification and flightproof vehicle testing.
8.3.6 Vehicle Vibration Test
The vibration test may be conducted instead of an acoustic test for small, compact vehicles that are
not sensitive to acoustic excitation. Such vehicles may be excited more effectively via interface
vibration. These vehicles should have a weight less than 400 lb.
8.3.6.1 Purpose
The vibration test demonstrates vehicle margin over the launch and ascent environments and assures
that a satisfactory workmanship level screen can be applied for follow-on hardware. Except for items
whose response is dominated by acoustic excitation, the vibration test also verifies the adequacy of
unit vibration qualification levels and serves as a qualification test and environmental stress screen for
items that have not been tested at a lower level of assembly.
10 3

8.3.6.2 Test Description
The vehicle and a flight-type adapter, in the ascent configuration, shall be vibrated using one or more
shakers through appropriate vibration fixtures. Vibration shall be applied in each of three orthogonal
axes, one direction being parallel to the vehicle thrust axis. Instrumentation shall be installed to
measure, in those same three axes, the vibration inputs and the vibration responses at attachment
points of critical and representative units.
8.3.6.3 Test Levels and Durations
The basic test levels and duration required for vehicles exposed to the liftoff and ascent vibration,
effective duration of 15 seconds (see 3.12), are as follows:
Qualification: 6 dB above acceptance for 3 min/axis
Protoqualification: 3 dB above acceptance for 2 min/axis
Acceptance: Envelope of the maximum predicted environment and minimum
workmanship level shown in Figure 8.3.7-1 for 1 min/axis
Notching is allowed based on impedance differences between the test and flight interface.
Test time should be divided approximately equally between redundant functions. When insufficient
test time is available at the full test level to test redundant circuits, functions, and modes, extended
testing using a spectrum no lower than 6 dB below the qualification spectrum shall be conducted as
necessary to complete functional testing.
Spectrum Values
Frequency (Hz) Minimum PSD (g2/Hz)
20 0.002
20 to 100 +3 dB per octave slope
100 to 1000 0.01
1000 to 2000 -6 dB per octave slope
2000 0.0025
The overall acceleration level is 3.8 grms.
Figure 8.3.7-1. Minimum random vibration spectrum, vehicle acceptance test.
10 4

| Spectrum Values |  |
| --- | --- |
| Frequency (Hz) | Minimum PSD (g2/Hz) |
| 20 20 to 100 100 to 1000 1000 to 2000 2000 | 0.002 +3 dB per octave slope 0.01 -6 dB per octave slope 0.0025 |
| The overall acceleration level is 3.8 grms. |  |

8.3.6.4 Supplementary Requirements
During the test, all electrical and electronic units used during launch, ascent, and on-orbit, or that may
be especially susceptible to vibroacoustic failure modes, shall be electrically energized and sequenced
through operational modes to the maximum extent practical. Continuous monitoring of appropriate
perceptive parameters, including input and output parameters, and the vehicle main bus by a power
transient-monitoring device, shall be provided to detect intermittent failures.
See B.1.5 for discussion of a damage-based approach to the analysis of flight vibration data for
determining the adequacy of established acceptance and qualification test requirements when new
flight data are outside the experience base used to derive the preflight MPE spectrum.
8.3.6.5 Option for Vehicle Random Vibration Testing
8.3.6.5.1 Flightproof Random Vibration Test
For program procurements of a single vehicle of less than 400 lb. weight, a flightproof random vibra-
tion test may be executed for vehicles meeting criteria of 8.3.6. This approach subjects the vehicle to
a flightproof test that is an enhanced acceptance test using protoqualification levels (+3 dB), over the
acceptance duration (one minute). Confidence is gained that each flightproof vehicle meets perfor-
mance requirements in flight after having successfully passed testing beyond the maximum expected
flight environments (MPE). Thus, flightproof testing is a check on the adequacy of each flight item,
considering build variability or defects introduced due to handling or testing. In addition, flightproof
reduces potential fatigue failures for retest due to the reduced exposure time. Since flightproof testing
does not validate an acceptance test program (see B.1.2), random vibration testing of any subsequent
vehicle shall be performed at flightproof levels and durations.
8.3.7 Vehicle Thermal Balance Test
8.3.7.1 Purpose
The thermal balance test provides the data necessary to verify the analytical thermal model and dem-
onstrates the ability of the vehicle thermal control subsystem to maintain specified temperature limits
of units for various operational scenarios throughout the entire vehicle. The thermal balance test can
be combined with the thermal vacuum test.
8.3.7.2 Test Description
The qualification or protoqualification vehicle shall be exposed to thermal environments expected by
the vehicle during its service life in a thermal balance test. Test instrumentation shall be installed that
produces data that can be correlated to the thermal model over the full range of seasons, equipment
duty cycles, ascent conditions, solar angles, maximum and minimum unit thermal dissipations,
including effects of bus voltage variations, and eclipse combinations. As a minimum, three test con-
ditions shall be imposed: a hot operational case, a cold operational case, and a cold non-operational
case. Two additional cases should be imposed: a transient case and a case chosen to check the valid-
ity of the correlated model. Other cases that are commonly simulated include eclipse, ascent, safe
mode, and “day-in-the-life” conditions.
Thermal balance test phases need not be worst-case-expected flight conditions, but they should not be
significantly different from these conditions. Special emphasis shall be placed on defining the test
10 5

conditions expected to produce the maximum and minimum temperatures of sensitive units such as
batteries. Sufficient measurements shall be made on the vehicle internal and external units to verify
the vehicle thermal design, hardware, and analyses. The operation and power requirements of all
thermostatically or electronically controlled heaters and coolers shall be verified during the test, and
appropriate control authority demonstrated, both on the primary and redundant circuits.
The test chamber, with the test item installed, shall provide a pressure of no higher than 13.3 mPa
(10–4 Torr) for space and upper-stage vehicles, or a pressure commensurate with service altitude for
launch vehicles. Where appropriate, provisions should be made to prevent the test item from “view-
ing” warm chamber walls, by using black-coated cryogenic shrouds of sufficient area and shape that
are capable of approximating liquid-nitrogen temperatures. The vehicle thermal environment may be
supplied by one of the following methods:
a. Absorbed Flux. The absorbed solar, albedo, and planetary irradiation is simulated using
heater panels or infrared (IR) lamps with their spectrum adjusted for the external thermal
coating properties, or using electrical resistance heaters attached to vehicle surfaces.
b. Incident Flux. The intensity, spectral content, and angular distribution of the incident solar,
albedo, and planetary irradiation are simulated.
c. Equivalent Radiation Sink Temperature. The equivalent radiation sink temperature is
simulated using infrared lamps and calorimeters with optical properties identical to those of
the vehicle surface.
d. Combination. The thermal environment is supplied by a combination of the above methods.
The selection of the method and fidelity of the simulation depends upon details of the vehicle thermal
design, such as vehicle geometry, the size of internally produced heat loads compared with those sup-
plied by the external environment, and the thermal characteristics of the external surfaces. Instru-
mentation shall be incorporated down to the unit level to evaluate total vehicle performance within
operational limits as well as to identify unit problems. The vehicle shall be operated and monitored
throughout the test. Dynamic flight simulation of the vehicle thermal environment should be pro-
vided unless the external vehicle temperature does not vary significantly with time.
Temperature measurement channels used to define thermal equilibrium shall have stabilities with
time (noise level evidenced by varying readings at constant temperature, including bias and precision)
commensurate with the time rate of change of temperature (dT/dt) used to define equilibrium. This
capability shall be defined as a requirement and demonstrated before test.
8.3.7.3 Levels and Duration
Test conditions and durations for the thermal balance test are dependent upon the vehicle configura-
tion, design, and mission details. Boundary conditions for evaluating the thermal control hardware
and design shall include the following:
a. Maximum external absorbed flux plus maximum internal dissipation
b. Minimum external absorbed flux plus minimum internal power dissipation
10 6

c. Minimum external absorbed flux plus minimum non-operating or stand-by power
dissipation
Temperature stabilization shall be achieved when the unit having the largest thermal time constant
has a temperature rate of change of less than 1°C measured over five hours. The thermal time con-
stant of the subsystems and mission profile both influence the time required for the vehicle to achieve
thermal equilibrium and, hence, the test duration.
8.3.8 Vehicle Thermal Vacuum Test
8.3.8.1 Purpose
The qualification thermal vacuum test demonstrates the ability of the vehicle to meet design require-
ments and establishes the thermal design margin and vehicle performance under thermal vacuum
conditions and temperature extremes. Acceptance thermal vacuum testing demonstrates the ability to
withstand the thermal stressing environment with margin on temperature range and number of cycles.
It also detects material, process, and workmanship defects that would respond to thermal vacuum and
thermal stress conditions.
8.3.8.2 Test Description
The vehicle shall be placed in a thermal vacuum chamber, and a performance test conducted to assure
readiness for chamber closure. The vehicle shall be divided into separate equipment zones based on
the thermal limits of the temperature-sensitive units and similar unit qualification temperatures within
each zone. Units that operate during ascent shall be operating and monitored for the effects of corona
and multipacting, as applicable, as the pressure is reduced to the lowest specified level. The rate of
chamber pressure reduction shall be no greater than during ascent, and may have to be slower to
allow sufficient time to monitor for the effects of corona and multipacting. Typically, this test does
not simulate launch depressurization; therefore, consideration shall be given to any hardware suscep-
tible to a rapid change in pressure. Equipment that does not operate during launch shall have electri-
cal power applied after the lowest specified pressure level has been reached. A thermal cycle begins
with the vehicle at ambient temperature. The temperature is raised to the specified high level and
stabilized. Following the high-temperature soak, the temperature shall be reduced to the lowest speci-
fied level and stabilized. Following the low-temperature soak, the vehicle shall be returned to ambi-
ent temperature to complete one thermal cycle. Performance tests shall be conducted during the first
and last thermal cycle at both the hot and cold temperature limits with functional operation and
monitoring of perceptive parameters during all other cycles. If simulation of the ascent environment
is desirable at the beginning of the test, the first cycle may begin with a transition to a cold thermal
environment rather than a hot thermal environment.
In addition to the thermal cycles for an upper-stage or space vehicle, the chamber may be pro-
grammed to simulate various orbital flight operations. Execution of operational sequences shall be
coordinated with expected environmental conditions, and a complete cycling of all equipment shall be
performed, including the operating and monitoring of redundant units and paths. Vehicle electrical
equipment shall be operating and monitored continuously throughout the test. Temperature monitors
shall assure attainment of temperature limits. Strategically placed witness plates, quartz-crystal
microbalances, or other instrumentation shall be installed in the test chamber to measure the outgas-
sing from the vehicle and test equipment.
10 7

Performance tests shall be conducted after unit temperatures have stabilized at the hot and cold tem-
peratures on the first and last cycle. Before the first cycle and following the last cycle, the test shall
also be performed at ambient. For any intermediate cycles, abbreviated performance tests at hot and
cold temperatures shall be performed. During these tests, electrical and electronic units, including all
redundant circuits and paths, shall be cycled through all operational modes. Perceptive parameters
shall be monitored for failures and intermittent conditions. All electrical circuits and all paths should
be verified for circuit performance and continuity. Performance tests may be conducted during tem-
perature transitions.
In some cases, vehicle performance and workmanship objectives may require a vehicle thermal cycle
test in addition to a vehicle thermal vacuum test. If payload performance or workmanship screening
requires a temperature range that cannot be achieved in ground vacuum testing, then a vehicle thermal
cycle test over the desired temperature range shall also be performed as an alternative strategy (A.1.1).
8.3.8.3 Test Levels and Durations
The vehicles shall be tested to the temperature ranges and cycles as shown:
Qualification: 10°C beyond acceptance temperatures for 8 cycles
Protoqualification: 5°C beyond acceptance temperatures for 4 cycles
Acceptance: Minimum and maximum predicted temperatures for 4 cycles
Temperatures in various equipment areas shall be controlled by the external test environment and
internal heating resulting from equipment operation. During the hot and cold half-cycles, the test
temperature is reached as soon as one unit in each equipment area is at its hot or cold temperature.
Temperature stabilization shall be achieved when the test article temperature is within the allowed
test tolerance on the specified test temperature and the temperature rate of change is less than 3°C per
hour. Unit temperatures shall not be allowed to go outside their applicable qualification, protoqualifi-
cation, or acceptance range at any time during the test. The pressure shall be maintained at no higher
than 13.3 mPa (10–4 Torr) for service above 100,000 meters, or the pressure commensurate with the
highest possible service altitude for lower altitudes.
The rate of temperature change shall equal or exceed the maximum predicted mission rate of change.
The thermal soak shall be at least eight hours at each temperature extreme during the first and last
cycles. For intermediate cycles, the thermal soak duration shall be at least four hours. Operating time
shall be divided approximately equally between primary and redundant units.
8.3.9 Mode Survey Test
Mode survey testing shall be conducted to obtain data required to develop or verify a dynamic model
for load analyses. It may be conducted as a system test, or a combination of subsystem tests, as
appropriate. Reference 17 defines the requirements for the test.
10 8

8.3.9.1 Purpose
The mode survey test is conducted to experimentally derive a structural dynamic model of a vehicle
or to provide a basis for test verification of an analytical model. After upgrading analytically to the
flight configuration, the model is used in the Verification Load Cycle defined in Reference 17. The
Verification Load Cycle loads are used to determine structural margins and the adequacy of the
structural static strength test (7.3.2). Mode survey tests are, therefore, critical for verification of
vehicle structural integrity and qualification of the structural subsystem as flight-ready. Where
practical, a mode survey test is also performed to define or verify models used in the final preflight
evaluation of structural dynamic effects on control subsystem precision, stability, and pointing
performance.
8.3.9.2 Test Description
The data obtained in the mode survey test shall be adequate to define the mode shapes, natural
frequencies, and damping values, for all modes of vibration that occur in the frequency range of
interest, typically up to 70 Hz. The first two modes in each lateral coordinate plane, the first axial
mode, and the first torsional mode shall be acquired, even if their frequencies lie outside the specified
test range. The quality of the measured modes shall be judged by computing the mass-weighted
orthogonality of the mode shapes, i.e., the off-diagonal terms of the unit normalized generalized mass
matrix should be equal to or less than 0.10.
8.3.9.3 Test Levels
The test is generally conducted at response levels that are low compared to the expected flight levels.
Limited testing shall be conducted to evaluate nonlinear behavior, with a minimum of three levels
used when significant nonlinearity is identified.
10 9

THIS PAGE INTENTIONALLY LEFT BLANK

9. Prelaunch Validation and Operational Tests
9.1 Prelaunch Validation Tests, General Requirements
Prelaunch validation testing is accomplished at the factory and at the launch base, with the objective
of demonstrating launch system and on-orbit system readiness. Prelaunch validation testing is usually
divided into two phases:
Phase A. Integrated system tests (Step 3 tests, Reference 2 as guidance)
Phase B. Initial operational tests and evaluations (Step 4 tests, Reference 2 as guidance)
During Phase A, the test series establishes the vehicle baseline data in the factory preshipment
acceptance tests. When the launch vehicle(s), upper-stage vehicle(s), and space vehicle(s) are first
delivered to the launch site, tests shall be conducted as required to assure vehicle readiness for inte-
gration with the other vehicles. These tests are intended to identify any changes that may have
occurred in vehicle parameters as a result of handling and transportation to the launch base. The
launch vehicle(s), upper stage vehicle(s), and space vehicle(s) may each be delivered as a complete
vehicle, or they may be delivered as separate stages and first assembled at the launch site as a com-
plete launch system. The prelaunch validation tests are unique for each program in the extent of the
operations necessary to ensure that all interfaces are properly tested. For programs that ship a com-
plete vehicle to the launch site, these tests primarily confirm vehicle performance, check for trans-
portation damage, and demonstrate interface compatibility.
During Phase B, initial operational tests and evaluations (Step 4 tests) are conducted following the
integrated system tests to demonstrate successful integration of the vehicles with the launch facility,
and that compatibility exists between the vehicle hardware, ground equipment, computer software,
and within the entire launch system and on-orbit system. The point at which the integrated system
tests end and the initial operational tests and evaluations begin is somewhat arbitrary since the tests
may be scheduled to overlap in time. To the greatest extent practical or as dictated by the procuring
agency, the initial operational tests in the launch vehicle and launch upper stage shall exercise every
operational mode in order to ensure that all mission requirements are satisfied. These Step 4 tests
shall be conducted in an operational environment, with the equipment in its operational configuration,
by the operating personnel, in order to test and evaluate the effectiveness and suitability of the hard-
ware and software. These tests should emphasize reliability, contingency plans, maintainability, sup-
portability, and logistics. These tests should assure compatibility with scheduled range operations
including range instrumentation.
9.2 Prelaunch Validation Test Flow
Step 4 testing (Reference 2 as guidance) of new or modified ground facilities, ground equipment, or
software should be completed prior to starting the prelaunch validation testing of the vehicles at the
launch base. The prelaunch validation test flow shall follow a progressive growth pattern to ensure
proper operation of each vehicle element prior to progressing to a higher level of assembly and test.
11 1

In general, tests should follow the launch base buildup cycle. As successive vehicles or subsystems
are verified, assembly proceeds to the next level of assembly. Following testing of the vehicles and
their interfaces, the vehicles are electrically and mechanically mated and integrated into the launch
system. Upper-stage vehicles and space vehicles employing a recoverable flight vehicle shall utilize a
flight vehicle simulator to perform mechanical and electrical interface tests prior to integration with
the flight vehicle. Following integration of the launch vehicle(s), upper-stage vehicle(s), and space
vehicle(s), performance tests of each of the vehicles shall be conducted to ensure its proper operation
following the handling operations involved in mating. Vehicle cleanliness shall be monitored. In
general, the Step 4 testing of the launch system is conducted first. Bus and Payload testing shall be in
accordance with appropriate verification and test plans.
9.3 Prelaunch Validation Test Configuration
During each test, the applicable vehicle(s) shall be in their flight configuration to the maximum extent
practical, consistent with safety, control, and monitoring requirements. For programs utilizing a
recoverable flight vehicle, the test configuration shall include any airborne support equipment required
for the launch, ascent, and space vehicle deployment phases. This equipment shall be mechanically
and electrically mated to the space vehicle in its launch configuration. All ground equipment shall be
validated prior to being connected to any flight hardware. Test provisions shall be made to verify
integrity of circuits into which flight jumpers, arm plugs, or enable plugs have been inserted.
9.4 Prelaunch Validation Test Descriptions
The prelaunch launch vehicle and upper stage validation tests shall exercise and demonstrate satis-
factory operation of each of the vehicles through all of their mission phases, to the maximum extent
practical. Test data shall be compared to corresponding data obtained in factory tests to identify
trends in performance parameters.
9.4.1 Performance Tests
Performance tests shall be conducted to validate integrated hardware and software performance and
flightworthiness. Mechanical tests shall be conducted for leakage, valve and mechanism operability,
and fairing clearance.
9.4.1.1 Simulators
When simulators are employed, the flight interfaces shall be validated when reconnected.
9.4.1.2 Explosive Ordnance and Non-Explosive Firing Circuits
Prior to final connection of the firing circuit to electro-explosive devices (EED) and non-explosive
actuators (NEA), the ignition energy levels and redundant circuit isolation shall be validated. Circuit
continuity and stray energy checks shall be made prior to connection of a firing circuit to ordnance
devices, and this check shall be repeated whenever that connection is opened and prior to
reconnection.
9.4.1.3 Transportation and Handling Monitoring
Monitoring for shock, vibration, temperature, and humidity shall be performed at a minimum at the
forward and aft interfaces between the shipping container transporter and the article being shipped,
11 2

and on the top of the article. Three-axis monitoring shall cover the entire shipment period and the
data evaluated as part of the receiving process.
9.4.1.4 Late Removal and Replacement of Flight Hardware
A performance test shall be performed when flight hardware or interfaces have been removed and/or
replaced. The performance test shall verify performance of all affected hardware and software and
potentially affected interfaces.
9.4.2 Propulsion Subsystem Leakage and Functional Tests
Performance tests of the launch vehicle propulsion subsystem(s) shall be conducted to verify the
proper operation of all units to the maximum extent practical.
9.4.3 Launch-critical Ground Support Equipment Tests
Hardware associated with ground subsystems that are flight critical and non-redundant (such as
umbilicals) shall have been subjected to appropriate tests under simulated functional and environ-
mental conditions of launch. These tests shall include an evaluation of radio frequency (RF) interfer-
ence between system elements, electrical power interfaces, and the command and control subsystems.
For further guidance on Range Safety see Reference 3.
On a new vehicle design or where significant design changes were made to the telemetry, tracking, or
receiving subsystem of an existing vehicle, a test shall be run on the first vehicle to ensure nominal
operation and that explosive ordnance devices do not fire when the vehicle is subjected to worst-case
electromagnetic interference environment.
9.4.4 Compatibility Test, On-Orbit System
9.4.4.1 Purpose
The compatibility test validates any required compatibility of the upper-stage vehicle, the space vehi-
cle, the on-orbit command and control network, and other elements of the space system.
9.4.4.2 Test Description
Facilities to perform system compatibility tests exist. These facilities can command the launch,
upper-stage, and space vehicles, and process telemetry from the vehicles, as well as perform tracking
and ranging, thus verifying the system compatibility, the command software, the telemetry processing
software, and the telemetry modes. The required tests shall include the following:
a. Verification of the compatibility of the radio frequencies and signal waveforms used
by the flight unit’s command, telemetry, and tracking links
b. Verification of the ability of the flight units to accept commands from the command
and control network(s)
c. Verification of the command and control network(s) capability to receive, process,
display, and record the vehicle(s) telemetry link(s) required to monitor the flight units
during launch, ascent, and on-orbit mission phases
11 3

d. Verification of the ability of the flight units to support on-orbit tracking as required
for launch, ascent, and on-orbit mission phases
e. Verification of all uplink and downlink command and telemetry paths or redundant
boxes as well as all keying material on-board the space vehicle and on the ground
f. Verification of all downlink frequencies that contain data
9.4.4.3 Supplementary Requirements
The compatibility test is made with every vehicle to verify system interface compatibility. The test
shall be run using the software versions that are integrated into the operational on-orbit software of
the vehicle under test. Following the completion of the compatibility test, the on-orbit command and
control network configuration of software, hardware, and procedures shall be frozen until the space
vehicle is in orbit and initialized.
9.5 Follow-On Operational Tests for Space Vehicles
9.5.1 Follow-On Operational Tests and Evaluations
Follow-on Operational Tests and Evaluations shall be conducted at the launch site in an operational
environment with the equipment in its operational configuration. The assigned operating personnel
shall identify operational system deficiencies. (See Reference 2.)
9.5.2 On-Orbit Testing
On-orbit testing should be conducted to verify the functional integrity of the space vehicle following
launch and orbital maneuvering. Other on-orbit testing requirements are an important consideration
in the design of any space vehicle. For example, there may be a need to calibrate on-line equipment
or to verify the operational status of off-line equipment while in orbit. However, on-orbit testing is
dependent on the built-in design features, and if testing provisions were not provided, the desired tests
cannot be accomplished. On-orbit tests are, therefore, so program peculiar that specific requirements
are not addressed in this Standard.
9.5.3 Reusable Flight Hardware
Tests of reusable flight hardware shall be conducted as required to achieve a successful space mis-
sion. Reusable hardware consists of the vehicles and units intended for repeated missions. Airborne
support equipment that performs its mission while attached to a recoverable launch vehicle is an
example of a candidate for reuse. The reusable equipment would be subjected to repeated exposure to
test, launch, flight, and recovery environments throughout its service life. The accumulated exposure
time of equipment retained in a recoverable vehicle and of airborne support equipment is a function
of the planned number of missions involving this equipment and the retest requirements between mis-
sions. The environmental exposure time of airborne support equipment is further dependent on
whether or not its use is required during the acceptance testing of other non-recoverable flight equip-
ment. In any case, the service life of reusable hardware should include all planned reuses and all
planned retesting between uses.
The test requirements for reusable space hardware after the completion of a mission and prior to its
reuse on a subsequent mission depend heavily upon the design of the reusable item and the allowable
11 4

program risk. For those reasons, specific details are not presented in this Standard. Similarly,
orbiting space vehicles that have completed their useful life spans may be retrieved by means of a
recoverable flight vehicle, refurbished, and reused. Based on present approaches, it is expected that
the retrieved space vehicle would be returned to the contractor’s factory for disassembly, physical
inspection, and refurbishment. All originally specified acceptance tests shall be conducted before
reuse.
11 5

THIS PAGE INTENTIONALLY LEFT BLANK

Appendix A. Thermal Test Considerations
A.1 Additional Thermal Test Considerations
A.1.1 Vehicle Alternate Thermal Strategy
In some cases, vehicle performance and workmanship objectives are better addressed in a vehicle
thermal cycle test than in a vehicle thermal vacuum test. If payload performance or workmanship
screening requires a temperature range that cannot be achieved in ground vacuum testing, then a
vehicle thermal cycle test over the desired temperature range shall also be performed.
When a vehicle thermal cycle test is performed, the number of required vehicle thermal vacuum
cycles may be reduced from the values given in 8.3.8.3. The test durations and levels of the vehicle
thermal cycling test (and the number of thermal vacuum cycles, performed per 8.3.8) shall be:
Qualification: 6 TC cycles over a 70°C minimum temperature range (and 4 TV
cycles over the temperature range specified in 8.3.8.3)
Protoqualification: 3 TC cycles over a 60°C minimum temperature range (and 2 TV
cycles over the temperature range specified in 8.3.8.3)
Acceptance: 3 TC cycles over a 50°C minimum temperature range (and 2 TV
cycles over the temperature range specified in 8.3.8.3)
The vehicle qualification thermal cycle test detects design defects and demonstrates the ability of the
vehicle to withstand the stressing environment associated with flight vehicle thermal cycle acceptance
testing, with a qualification margin on temperature range and maximum number of cycles. The vehi-
cle acceptance thermal cycle test detects material, process, and workmanship defects. The vehicle
shall be placed in a thermal chamber at ambient pressure, and a performance test shall be performed
to assure readiness for the test. The vehicle shall be operated and monitored during the entire test,
except that vehicle power may be turned off, if necessary, to reach stabilization at the cold tempera-
ture. Vehicle operation shall be asynchronous with the temperature cycling, and redundant units shall
be operated for approximately equal times.
Temperature cycling shall begin when the relative humidity of inside spaces of the vehicle is below
the value at which the cold test temperature would cause condensation. One complete thermal cycle
is a period beginning at ambient temperature, then cycling to one temperature extreme and stabilizing,
then to the other temperature extreme and stabilizing, and then returning to ambient temperature.
Strategically placed temperature monitors installed on units shall assure attainment and stabilization
of the expected temperature extremes for all units. Auxiliary heating and cooling may be employed
for selected temperature-sensitive units (e.g., batteries). If it is necessary in order to achieve the
required temperature rate of change, parts of the vehicle such as solar arrays and passive thermal
equipment may be removed for the test.
11 7

Performance tests shall be conducted after unit temperatures have stabilized at the hot and cold tem-
peratures on the first and last cycle. Before the first cycle and following the last cycle, the perform-
ance test shall also be performed at ambient. For intermediate cycles, functional tests at hot and cold
temperatures shall be performed. During these tests, electrical and electronic units, including all
redundant circuits and paths, shall be cycled through all operational modes. Perceptive parameters
shall be monitored for failures and electrical intermittences. All electrical circuits and all paths shall
be verified for circuit performance and continuity. Performance tests and mission profile testing may
be conducted during temperature transitions.
A.1.2 Option for Unit Level Two-Tier Thermal Test
When an electrical or electronic unit’s allowable test temperature limits do not comply with the
baseline temperature ranges specified in 6.3.8.3b and 6.3.9.3b, the number of cycles may be increased
to achieve an equivalent screening stress level. An alternative approach is applicable when opera-
tional temperature limits can comply with the baseline temperature ranges, but performance temper-
ature limits cannot. In such cases, a two-tier thermal test profile may be adopted whereby the unit
demonstrates operational requirements at the baseline temperature range and performance require-
ments at a narrower temperature range. A representative two-tier acceptance thermal test profile is
shown in Figure A.1.2.
For an acceptance test, the unit shall be taken to the acceptance temperature range (at least -24°C to
+61°C) on each cycle of the test and to a narrower performance temperature range on the first and last
cycles. Functional tests shall be conducted at acceptance temperatures on each cycle. Performance
tests are conducted at the narrower performance hot and cold temperature levels on the first and last
cycle. Other test requirements, such as dwell time and parameter monitoring during the test, are as
specified in 6.3.8 and 6.3.9.
Figure A.1.2. Two-tier acceptance thermal test profile.
11 8

An alternative approach is to conduct performance testing at the acceptance levels with the provision
that performance requirements do not need to be met at these wider temperature levels. When the
unit is ramped to the subsequent performance temperature range, only those performance tests that
did not pass at the acceptance temperature are conducted. Performance testing at the acceptance tem-
perature range permits some performance requirements to be verified at the wider temperature range
and a better understanding of performance roll-off or degradation as a function of temperature.
Another option is to conduct all performance testing at both the acceptance and the performance tem-
perature levels, allowing performance to degrade at the acceptance level. The advantage of this
approach is that it characterizes the unit’s performance temperature dependency.
In two-tier testing, the thermal uncertainty margin is demonstrated between the maximum thermal
model temperature prediction range and the performance temperature limits. The thermal control
subsystem shall be designed (i.e., selection of thermal control hardware, heaters and thermal coatings)
to show, for an acceptance unit, an 11°C minimum margin between the maximum model temperature
prediction and the hot performance temperatures, and an 11°C minimum margin (or a 25 percent
control authority for active thermal control) between the minimum model temperature prediction and
the cold performance temperatures. For protoqualification testing, the thermal control subsystem
shall show an additional 5°C of temperature margin.
11 9

THIS PAGE INTENTIONALLY LEFT BLANK

Appendix B. Dynamic Test Considerations
B.1 Test Considerations for Acoustic, Vibration, and Shock Environments
The following paragraphs describe the considerations for the statistical analysis approach used in this
document to derive the Maximum Predicted Environment for acceptance testing and margins for
qualification and protoqualification. The approach applies to random vibration, acoustics, and shock
environments.
B.1.1 Statistical Basis for Test Level
a. Flight-to-flight variability of the spectral value at a frequency for acoustic, random
vibration, shock, and sinusoidal vibration environments (defined in 3.26, 3.27, 3.28,
and 3.29 respectively) is baselined to be log-normally distributed. That is, the normal
distribution applies to the logarithms of the spectral values at a particular frequency.
Consequently, the estimated mean spectrum is the average of the logarithmic values
of available spectra. The standard deviation of spectra from the mean is denoted by
and is baselined to equal 3 dB. The assumption of log-normal distribution with 3 dB
standard deviation is based on repeated measurements on 24 static firings and over 4σ0
flights of a launch vehicle (Reference B9).
b. Test levels are generally based on a tolerance interval above the estimated flight
mean spectrum. The interval is specified for a probability P that the test spectrum
will not be exceeded in flight, estimated with a confidence of C, and will depend on
the applicable probability distribution. In the absence of a set of relevant measure-
ments, the analysis presented in the next several paragraphs may be used. It rests on
the assumption that the distribution of the spectrum values is normal, centered at the
sample or analysis mean, and that the standard deviation of the overall population
from which the sample was taken is 3 dB as discussed in item (a) above. This
approach has been used in versions of MIL-STD-1540 since 1989. More recently,
Womack (Reference B2) documented the applicability of the analysis and set the low
sample limit at approximately 14. For larger samples, a rigorous calculation of the
tolerance interval should be employed. Reference B10 provides guidance for such
cases.
For the special case where σ is known and the mean spectrum value is estimated
from N available flights, the test level L for probability P with confidence C is given
by
LP/C = [
P
+ /N1/2] dB (B.1)
C
The factor multiplied by σd eztermzines the normal probability limit, and the factor
P
/N1/2 multiplied by determines the confidence limit for the estimate of the mean
C
spectrum. zThe factors anσd are read from a table of the standardized normal
P C
dzensity function founσd in many references, such as Reference B3. On a normal
distribution plot, the areza P liezs to the left of the mean plus times the standard
P
z
12 1

deviation . Likewise, the area C lies to the left of the mean plus times the stand-
C
ard deviation . Note that = 0 for 50% confidence since the mean is the 50-
C
percentile σestimate. Numerical examples for acceptance, qualificaztion, and pro-
toqualificationσ are includedz in c, d, and e below.
c. Acceptance tests are performed at the P95/50 level, with consideration of the mini-
mum workmanship level, shorthand for the probability P = 0.95 and the confidence C
= 0.50. Stated another way, there is a 50-50 chance of one exceedance of the P95/50
spectrum in 20 flights. Reading from the following table that = 1.645 and =
0, acceptance is performed at 4.9 dB above the mean spectrum [from Eq. (B.1), L95/50
0.95 0.50
= 3(1.645 + 0) = 4.9 dB]. Note that this value applies no mattezr ho w many flighzts
provide data for the estimate of the mean. Also note that minimum spectra from Fig-
ures 6.3.5-1, 8.3.6-1, and 6.3.6-1 must be enveloped with the P95/50 spectrum for the
final acceptance test levels for random vibration and acoustic tests.
d. Qualification is performed at the P99/90 level (P = 0.99 and C = 0.90). Stated
another way, there is one chance in ten of exceeding the qualification level once in
100 flights. For the purpose of preflight prediction, a value N = 1 is adopted. Then,
since = 2.322 and = 1.282, preflight qualification is performed at 10.8 dB
above the mean spectrum [from Eq. (B.1), L99/90 = 3(2.322 + 1.282) = 10.8 dB]. The
0.99 0.90
prefligzht q ualification szpect rum is therefore baselined to be 6 dB above the
acceptance spectrum (a rounding of 10.8 – 4.9 = 5.9). The 6-dB qualification margin
is the same as in versions A and B of this MIL-STD-1540, where it was based on
experience and not on a statistical model.
After N flights are available, updates of the estimates can be made as follows:
1. A revised estimate of the mean spectrum is calculated.
2. The revised estimate of the L95/50 is then 4.9 dB above the revised mean. The
result is compared to the previously established acceptance spectrum.
3. The revised estimate of the qualification test margin M, the dB difference
between the qualification and acceptance levels (from paragraphs c and d
above), is
M = L99/90 L95/50 = 3(2.322 + 1.282 / N1/2) 3(1.645 + 0)
= 3(0.677 + 1.282 / N1/2) dB (B.2)
− −
As the number N of flight data samples grows, the margin decreases since the confi-
dence in its estimate increases. The following table summarizes the estimated margin
to the number of flights using the relationship of Eq. (B.2):
Table B.1.1-1. Qualification Margin vs. Number of Flights
Number of flights, N 1 2 3 4 5 6 8 12
Margin, M (dB) 5.9 4.8 4.3 4.0 3.8 3.6 3.4 3.2
12 2

| Number of flights, N | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 12 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Margin, M (dB) | 5.9 | 4.8 | 4.3 | 4.0 | 3.8 | 3.6 | 3.4 | 3.2 |

The difference is not allowed to fall below 3 dB. When data from a sufficient num-
ber of flights, or from ground tests that produce environments that are realistic for
flight (for example, engine firings), the tolerance interval of the applicable statistical
distribution can be determined using the available data. However, when reducing the
qualification margin on the basis of improved confidence in the environment, one
must not put at risk the peak stress capability and fatigue life demonstration that
comes from the qualification test nor should the margin drop below the maximum
allowable tolerance for vibration testing, currently ± 3.0 dB, as indicated in Table
4.7-1.
e. Protoqualification is performed at 3 dB above the 95/50 acceptance level, an estab-
lished practice set to be half the 6 dB increase for baseline qualification. Assuming
that the assumed statistical distribution is valid, the protoqualification spectrum is
7.9 dB above the mean (3 dB over the 4.9 dB for acceptance). Since 7.9 dB is 2.63
sigma above the mean (7.9/3), with 50% confidence the probability of exceeding the
protoqualification level in flight at any particular frequency is 0.0043 (about 1 in
230); this result is read as the probability of exceeding 2.63 sigma above the mean in
a normal density table (Reference B3). For a 90% confidence, the probability of
exceeding the protoqualification level for a single flight is 0.088 or exceedance will
occur once in 11 flights; this result is obtained from Eq. (B.1), 7.9 = 3[ + 1.282/1]
yielding = 1.351 and resulting in a probability of exceedance of 0.088 read from a
P
normal density table. z
P
z
B.1.2 Acceleration of Acceptance Life for Acoustic and Random Vibration Tests
Spacecraft and many launch vehicle components are exposed to acoustics and random vibration during
the liftoff and ascent segments of flight for a nominal period of 15 seconds. Some components may be
exposed to these environments in excess of 15 seconds, such as those located on or near engines.
Baseline acoustic and random vibration qualification and protoqualification tests include one minute
duration for the liftoff and ascent flight environment with a margin added to the acceptance spectrum.
A longer than the baseline 15-second duration of the maximum predicted environment (see 3.12)
leads to an increased test time for flight of four times that of the MPE, where four is the duration fac-
tor for fatigue life demonstration by test. To ensure that flight capability is maintained after the
acceptance program on production hardware, the test duration is increased beyond the time required
for flight to serve as a life test for a maximum duration acceptance testing. The assumptions are that
fatigue is the life-limiting mechanism, that Miner’s Rule for fatigue accumulation applies, and that
induced stress is proportional to the applied acceleration. Miner’s Rule (Reference B6) states that the
summation of the product of the number of cycles times their stress amplitude raised to an exponent
“b” is proportional to the fraction of life exhausted. Therefore, if T denotes the upper limit on the
A
duration of acceptance testing, 4T becomes the duration of the life test for acceptance required if
A
performed with the acceptance spectrum.
Since the qualification and protoqualification testing are performed at higher than the acceptance
level beyond the duration required for flight, the added testing becomes an accelerated acceptance life
test. The time acceleration factor is given by the amplitude factor on the acceptance excitation raised
to the fatigue exponent “b.” The amplitude factor equals 10M/20, where M is the margin in dB. So the
12 3

time acceleration factor is 10Mb/20. Let t be the duration of an acceptance test (baseline 1 minute), T
A A
be the limit on the duration of acceptance testing, and 4 be the life factor, then
T / t = (1/4)10Mb/20 (B.3)
A A
For conservatism, the exponent on stress is taken to be 4, a conservative value for this purpose. For
example, Reference B3 recommends b = 4 for solder.
T / t = (1/4)10M/5 (B.4)
A A
A table of the acceptance duration limit versus the test margin M follows:
Table B.1.2-1. Allowable Acceptance Test Duration for Various Test Margins
Test margin, M (dB) 3 4 4.5 5 6
Acceptance limit, TA / tA 1.0 1.6 2.0 2.5 4.0
As seen above, one minute of 6 dB margin testing demonstrates life for four acceptance tests of one
minute each. Since baseline qualification uses a 6 dB margin and a three-minute test (two minutes
beyond the one min for flight), adequate remaining life for flight life is demonstrated for up to eight
1one-minute acceptance tests. Note that each minute with a 3 dB margin demonstrates life for a
single acceptance test. So, for protoqualification (3 dB margin for two minutes, one of which is for
flight), a limit of only one acceptance test is demonstrated. Therefore, under nominal assumptions,
there is no demonstrated life remaining to accommodate any retesting.
B.1.3 Margin and Retest Implications of Acoustic and Random Vibration
Qualification and Protoqualification Tests
In general, the test margin for qualification or protoqualification is M dB over acceptance, and the
upper bound on acceptance testing (per axis for vibration) is T . Based on B.1.2, the general
A
requirement for the duration of a qualification or protoqualification test is given by
T = 4(T + T /10Mb/20) (B.5a)
Q MPE A
This equation includes the implicit assumption that maximum vibration levels in flight could be as
high as the qualification (or protoqualification) environment and, as a result, the flight duration, T ,
MPE
is not accelerated. This is a conservative approach justified by the uncertainty in the true flight envi-
ronment for future flights.
The nominal qualification strategy employs a three minute per axis test at 6 dB above the acceptance
level. The first goal of the qualification test is to demonstrate that the design is robust to the qualifi-
cation environment. Under the assumption that hardware is designed to survive the full qualification
test level, the qualification test hardware is generally not usable without substantial inspection and
rework. The subsequent item is typically the first flight hardware. For this item, the life margin of 4
(Eq. B.3) implies that for a nominal 15 second exposure, one minute of the qualification test
demonstrates margin for flight, at the qualification or P99/90 level. Table B.1.2-1 shows that the
12 4

| Test margin, M (dB) | 3 | 4 | 4.5 | 5 | 6 |
| --- | --- | --- | --- | --- | --- |
| Acceptance limit, TA / tA | 1.0 | 1.6 | 2.0 | 2.5 | 4.0 |

remaining two minutes equate to the fatigue experienced in eight one-minute acceptance tests: one for
the expected acceptance test and seven more for retest after rework or repair.
In contrast, the nominal protoqualification strategy employs a two-minute-per-axis test at 3 dB above
the acceptance level. Referring again to Table B.1.2-1, this test demonstrates fatigue life for two one-
minute acceptance tests. The protoqualification unit is flown at risk, relying on analytical margin
rooted in confidence in the design and manufacturing processes. The second flight unit then has
demonstrated fatigue life for one acceptance test and one flight exposure at the P95/50 level. Life has
not been demonstrated for any repeat of any original acceptance test.
An alternative test approach that meets the acceptance life demonstration with less conservatism is
described in B.1.4.
In those cases where the hardware is exposed to vibration at or near acceptance levels for long dura-
tions, such as engine components, Eq. (B.5a) is modified to accelerate both flight and total acceptance
test durations
T = 4(T + T )/10Mb/20 (B.5b)
Q MPE A
to preclude excessive fatigue accumulation in test. For this approach, the qualification of the affected
hardware to demonstrate capability to withstand qualification levels of vibration is performed as a
separate test. The duration of that test is determined on a case-by-case basis depending on the appli-
cation and mission requirements.
B.1.4 Two-Phase Qualification and Protoqualification Test for Vibration
and Acoustics
The discussion in the previous section can be used as the basis for a modified test strategy where the
demonstration of margin to test level and duration for acceptance testing is separated into different
phases of the test. The testing consists of a Phase I for acceptance life performed with the acceptance
spectrum and a Phase II for flight with the qualification or protoqualification spectrum. The two-
phase test approach can be employed to
a. Reduce the conservatism in the testing for acceptance life that is built into the baseline
qualification and protoqualification requirements.
b. In those cases where the hardware is exposed for long durations at or near acceptance levels
of vibration.
c. Testing of hardware that is vibration or shock isolated in flight, but acceptance tested without
isolators.
In baseline testing, the acceptance life test is accelerated since it is performed at levels higher than
acceptance. As indicated earlier, the acceleration of the life test is based on a nominal exponent of 4
12 5

for fatigue, as well as the assumptions of linearity and that all amplitudes contribute to fatigue life
(that is, not allowing for amplitudes below an endurance limit). For example, by performing the
acceptance life testing at 6 dB higher than acceptance (a factor of 2 in amplitude), a time acceleration
factor of 24 or 16 is used. If the fatigue exponent were 6, the time acceleration factor would be 26 or
64.
For qualification, Figure B.1.1 depicts the baseline approach, consisting of testing at 6 dB above
acceptance for three minutes.
A corresponding two-phase test is conducted for 32 minutes at the acceptance level, Phase I, followed
by a one-minute test at 6 dB above the acceptance level, Phase II. Figures B.1.2a and B.1.2b provide
the graphical representation. Phase I of the testing is an acceptance life test consisting of a normal
acceptance test extended in duration to 4 times the set limit on the duration of flight acceptance
testing (T ). For example, a 32-minute Phase (per axis for vibration) covers a baseline maximum of
FLT
eight minutes of acceptance testing. Phase II is a baseline qualification test for a duration of 4 times
the effective flight duration but not less than 1 min (4.3.2.2) at 6 dB above acceptance
Figure B.1.1. Qualification test (+6 dB, 3 minutes).
Figure B.1.2a. Phase I qualification test for equivalent acceptance life (0 dB, 32 minutes).
12 6

Figure B.1.2b. Phase II test for qualification margin (+6 dB, 1 minute).
For protoqualification, Phase I is an acceptance life test consisting of a normal acceptance test
extended in duration to four times the set limit on the duration of flight acceptance testing, the same
requirement as for qualification. Phase II is a baseline protoqualification test for a duration of four
times the effective duration of the flight acceptance testing (T ), but not less than one minute
FLT
(4.3.3.2). So the only change from qualification is the margin used to qualify the hardware, which
now is 3 dB. This test approach is shown graphically in Figures B.1.3, B.1.4a, and B.1.4b.
Figure B.1.3. Protoqualification test (+3 dB, 2 minutes).
Figure B.1.4a. Phase I protoqualification test for equivalent acceptance life (0 dB, 4 minutes)
and possible extension for n lives.
12 7

Figure B.1.4b. Phase II test for protoqualification margin (+3 dB, 1 minute).
The accelerated acceptance life segment of the protoqualification test is one minute at 3 dB. The
equivalent Phase I duration is four minutes. This part of the test can be extended to demonstrate
additional retest capability. Each additional four minutes of testing demonstrates a one minute retest
capability. This added flexibility in Phase I testing allows the tailoring of the two phase test to suit
program requirements without exposing the hardware to the higher levels of testing at 3 dB above
acceptance. Phase II testing is a one-minute test at 3 dB above acceptance level, demonstrating
protoqual level capability of the hardware.
When performing two phase testing, the sequence of Phase I and II tests can be tailored to address a
program-unique issue. In general, the test phase least likely to damage the hardware should be per-
formed first. Acceptance level testing should be performed first if hardware is more sensitive to peak
stress. Qualification, or protoqualification, testing should be performed first if the hardware is sensi-
tive to fatigue damage.
B.1.5 Damage-Based Analysis of Flight Vibroacoustic Data
Traditional maximax spectral analysis of flight vibroacoustic data for space and launch vehicles (3.15
and 3.20) can lead to excessively conservative testing. An alternate data analysis method (Reference
B5), based on a simple damage model, employs an extended response spectrum analysis that includes
amplitude-cycle counts to deal with fatigue potential. The output is a conservative stationary test
specification, but less so than using the maximax basis. The damage-based test specification
envelops the damage potential of the non-stationary flight environment for both peak response and
fatigue, while recognizing uncertainties in damping and in the fatigue law.
The advanced method enables a more perceptive means for assessing the flightworthiness of units
when maximax analysis of new flight data indicates excessive levels. Re-qualification or vibration
isolation may then be required. Some experience with the advanced data analysis technique indicates
a potential to clear the concern.
B.1.6 Threshold Response Spectrum for Shock Significance
The damage potential of a shock test may be shown to be less than the damage potential from the ran-
dom vibration acceptance testing over its frequency range, typically 20 to 2000 Hz. For the shock
response spectrum values, a modal response velocity criterion can be used to signify a lack of shock
12 8

severity, as long as the unit does not contain any components that are sensitive to shock, such as
crystals and ceramic chips (6.3.4.4).
The response spectrum S of the random vibration acceptance excitation is given in Reference B4 as,
vib
S
vib
= n[(π/2)GvibfQ]1/2 (B.6)
n = factor on the response standard deviation to yield maximum response
G = spectral density of random vibration (g2/Hz)
vib
f = frequency (Hz)
Q = quality factor
An expression for n with 50% confidence is given in Reference B4 as
n = [2 ln(fT)]1/2 (B.7)
where ln is the natural logarithm and T is the duration of the random test. For a 60-second random
vibration test, n is 3.8 at 20 Hz increasing to 4.8 at 2000 Hz. Substituting Eq. (B.7) into Eq. (B.6),
S = [πGfQ ln(fT)]1/2 = 5.6[Gf ln(fT)]1/2 for Q = 10 (B.8)
vib
If S exceeds the response spectrum for the shock S at all frequencies, the random vibration test
vib shock
is judged to have a damage potential greater than that of the shock over the frequency range of the
vibration. A response velocity to shock less than 50 in/sec is judged to be non-damaging (Reference
B7). This is the case if the shock response spectrum value in g is less than 0.8 times the frequency in
Hz.
See 6.3.4 for qualification test exception requirements.
B.2 Response Limiting Criteria for Units Weighing More Than 50 lb (23 kg)
Force or response limiting refers to the practice of notching (reduction of level in frequency bands) of
the input acceleration spectrum to a test item to reduce either the applied force spectrum or to reduce
the magnitude of the spectrum of test item response at critical locations. In both cases, the reduction
is in frequency bands that contain major resonant behavior of the test item or of the test fixture. A
justifiable basis for such limiting is necessary in order to avoid excessive reduction of inputs that will
result in inadequate acceptance or qualification.
The rationale is that the input motion is reduced because the relatively high mechanical impedance of
the test item inhibits the motion of the supporting structure that would occur in the flight configura-
tion and because the test specifications are based on enveloping response data.
12 9

B.2.1 Broadband Reduction
For units exceeding 23 kg (50 lb), the random vibration specification may be reduced using the fol-
lowing relation:
Reduced spectrum level (g2/Hz) = (B.9)
where W is the unit weight in pounds. The reduction cannot be more than 6 dB. Figure B.2.2-1
shows the minimum spectra for units weighing 50, 100, and 200 lb, respectively. For each of these
weights, the flat portion of the spectrum was extended into the low-frequency regime without reduc-
ing the spectrum roll-off level in order to assure adequate excitation of the lower frequency modes
resulting from the increased weight.
B.2.2 Narrowband Notching
The input vibration spectrum may be adjusted on a case-by-case basis to limit unit response accelera-
tions. The adjustment takes the form of notching of the input over a narrow frequency band. The
notch depth is limited to 10 dB of below the input PSD, with floor not less than 0.01 g2/Hz. Figure
B.2.2-2 shows an example of notching.
In addition, the notch bandwidth shall not exceed ±5% of the notch center band frequency.
As an example, for 200 Hz center band notch frequency, f , the notch band is ±10 Hz, i.e., 190 Hz
c
minimum and 210 Hz maximum.
B.3 Anomaly Severity Definitions
Dhallin and Graham define the severity of Category 1, 2, and 3 ground and flight anomalies in
Reference B8. In summary:
Category 1: Loss of mission or permanent inability to perform baseline mission
Category 2: Permanent loss of redundancy or loss of reliability; temporary inability of the
spacecraft or an operational payload to meet its baseline mission requirements
Category 3: Anomalies that are identified as a nuisance or of negligible impact. This includes
anomalies with non-operational payloads: recorded flight observations that have no vehicle
impact.
13 0

Weight (lb) Overall Acceleration (Grms)
50 6.90
100 4.87
200 3.52
Figure B.2.2-1. Weight-adjusted minimum random vibration spectrum, unit acceptance test.
Figure B.2.2-2. PSD notch.
13 1

| Weight (lb) | Overall Acceleration (Grms) |
| --- | --- |
| 50 100 200 | 6.90 4.87 3.52 |

B.4 References
B1. Pendleton, L. R. and Henrikson, R. L., Flight-to-Flight Variability in Shock and Vibration
Levels Based on Trident I Flight Data, Proceedings of the 53rd Shock and Vibration Sympo-
sium, 1983.
B2. Womack, J. M., Statistical Tolerance Bounds: Overview and Application to Spacecraft and
Launch Vehicle Dynamic Environments, 28th Aerospace Testing Seminar, Los Angeles, CA,
March 2014.
B3. Bendat, J. S. and Piersol, A. G., Random Data Analysis and Measurement Procedures, 3rd
edition, John Wiley & Sons, Inc., New York, 2000.
B4. Steinberg, D. S., Vibration Analysis for Electronic Equipment, 3rd edition, John Wiley &
Sons, Inc., New York, 2000.
B5. DiMaggio, S. J., Sako, B. H., and Rubin, S., Analysis of Nonstationary Vibroacoustic Flight
Data Using a Damage-Potential Basis, AIAA Dynamic Specialists Conference, 2003 (also,
Aerospace Report No. TOR-2002(1413)-1838, 1 August 2002).
B6. Harris, C. M. and Piersol, A. G., Shock and Vibration Handbook, Fifth edition, Chapter 34,
McGraw-Hill, New York, pp. 34.17-34.22, 2002.
B7. Gaberson, H. and Chalmers, R., Modal Velocity as a Criterion of Shock Severity, Proceed-
ings of the 40th Shock and Vibration Symposium, Dec. 1969.
B8. Dhallin, A. and Graham, S., Findings and Lessons Learned from Operational Anomaly
Trending and Analysis, 27th Aerospace Testing Seminar, Los Angeles, CA, Oct. 2012.
B9. Odeh, R. E. and Owen, D. B., Tables for Normal Tolerance Limits, Sampling Plans and
Screening, Marcel Decker Inc., June 1980.
13 2

SMC Standard Improvement Proposal
INSTRUCTIONS
1. Complete blocks 1 through 7. All blocks must be completed.
2. Send to the Preparing Activity specified in block 8.
NOTE: Do not use this form to request copies of documents, or to request waivers, or clarification of requirements on
current contracts. Comments submitted on this form do not constitute or imply authorization to waive any portion of
the referenced document(s) or to amend contractual requirements. Comments submitted on this form do not
constitute a commitment by the Preparing Activity to implement the suggestion; the Preparing Authority will
coordinate a review of the comment and provide disposition to the comment submitter specified in Block 6.
SMC STANDARD 1. Document Number 2. Document Date
CHANGE
SMC-S-016 5 September 2014
RECOMMENDATION:
3. Document Title TEST REQUIREMENTS FOR LAUNCH, UPPER-STAGE AND SPACE
VEHICLES
4. Nature of Change
(Identify paragraph number; include proposed revision language and supporting data. Attach extra sheets as needed.)
5. Reason for Recommendation
6. Submitter Information
a. Name b. Organization
c. Address d. Telephone
e. E-mail address 7. Date Submitted
8. Preparing Activity Space and Missile Systems Center
AIR FORCE SPACE COMMAND
483 N. Aviation Blvd.
El Segundo, CA 91245
Attention: SMC/EN
February 2013

| SMC Standard Improvement Proposal |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| INSTRUCTIONS 1. Complete blocks 1 through 7. All blocks must be completed. 2. Send to the Preparing Activity specified in block 8. NOTE: Do not use this form to request copies of documents, or to request waivers, or clarification of requirements on current contracts. Comments submitted on this form do not constitute or imply authorization to waive any portion of the referenced document(s) or to amend contractual requirements. Comments submitted on this form do not constitute a commitment by the Preparing Activity to implement the suggestion; the Preparing Authority will coordinate a review of the comment and provide disposition to the comment submitter specified in Block 6. |  |  |  |  |  |  |
|  | SMC STANDARD |  |  | 1. Document Number SMC-S-016 |  | 2. Document Date 5 September 2014 |
|  | CHANGE |  |  |  |  |  |
|  | RECOMMENDATION: |  |  |  |  |  |
| 3. Document Title |  | TEST REQUIREMENTS FOR LAUNCH, UPPER-STAGE AND SPACE VEHICLES |  |  |  |  |
| 4. Nature of Change (Identify paragraph number; include proposed revision language and supporting data. Attach extra sheets as needed.) |  |  |  |  |  |  |
| 5. Reason for Recommendation |  |  |  |  |  |  |
| 6. Submitter Information |  |  |  |  |  |  |
| a. Name |  |  |  |  | b. Organization |  |
| c. Address |  |  |  |  | d. Telephone |  |
| e. E-mail address |  |  |  |  | 7. Date Submitted |  |
| 8. Preparing Activity Space and Missile Systems Center AIR FORCE SPACE COMMAND 483 N. Aviation Blvd. El Segundo, CA 91245 Attention: SMC/EN |  |  |  |  |  |  |
