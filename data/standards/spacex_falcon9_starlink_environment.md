---
standard: SpaceX Falcon 9 / Starlink Gen2 Environment Specification
method: LEO_RAD_VIB
category: radiation_vibration
language: ko
source: SpaceX Falcon 9 Users Guide Rev 2.0 / Starlink Gen2 Mission Overview
---

# SpaceX Falcon 9 / Starlink Gen2 환경 규격

## 1. 임무 개요

Starlink Generation 2 Mini 위성은 SpaceX가 운용하는 저궤도(LEO) 광대역 위성 인터넷 서비스용 위성이다.
- 궤도 고도: 550 km 극궤도(inclination ≈ 53°)
- 설계 수명: 5년 (임무 종료 후 대기권 재진입)
- 위성 질량: ≈ 800 kg (Gen2 Mini 기준)
- 총 운용 위성 수: 6,000기 이상 (2024년 기준)

---

## 2. 방사선 환경 (Radiation Environment)

### 2.1 총이온화선량 (TID)

550 km LEO 궤도에서 Al 4mm 차폐 기준:

| 항목 | 수치 | 단위 | 비고 |
|---|---|---|---|
| 연간 TID(Solar Minimum) | 2.0 | krad(Si)/yr | 지자기 차폐 효과 포함 |
| 연간 TID(Solar Maximum) | 4.0 | krad(Si)/yr | 태양 양성자 이벤트 포함 |
| 5년 누적 TID | 10 ~ 20 | krad(Si) | 궤도 경사각·차폐 두께에 따라 편차 |
| **설계 요구치(마진 포함)** | **≥ 15** | **krad(Si)** | **RDM(Radiation Design Margin) 1.5× 적용** |

> 규격 기준: SpaceX Starlink Gen2 LEO 5년 TID 요구치 **15 krad(Si)** 이상.

### 2.2 단일이벤트효과 (SEE)

| 항목 | 임계값 | 비고 |
|---|---|---|
| SEL LET 임계치 | > 37 MeV-cm²/mg | Starlink 운용 환경 최대 LET |
| 권장 SEL 면역 LET | > 60 MeV-cm²/mg | 설계 마진 포함 |
| SEU 발생률 | < 10⁻⁷ errors/bit/day | 비트 오류 허용 한도 |

---

## 3. 발사 진동 환경 (Launch Vibration — Falcon 9)

### 3.1 랜덤 진동 (Random Vibration)

Falcon 9 Payload Attach Fitting(PAF) 인터페이스 기준:

| 주파수 대역 | PSD (g²/Hz) | 비고 |
|---|---|---|
| 20 – 100 Hz | 0.005 | 저주파 롤오프 |
| 100 – 200 Hz | +6 dB/oct | 상승 구간 |
| 200 – 700 Hz | 0.02 | 최대 PSD 구간 |
| 700 – 2000 Hz | -6 dB/oct | 하강 구간 |
| **종합 g rms** | **≈ 8.8 g rms** | **20 – 2000 Hz 적분값** |
| 지속 시간 | 120 sec/axis | 3축 |

> 규격 기준: Falcon 9 PAF 기준 랜덤 진동 내구 요구치 **8.8 g rms** 이상.  
> 출처: Falcon 9 Users Guide Rev 2.0, Table 4-5 (Random Vibration Environment)

### 3.2 정현파 진동 (Sine Vibration)

| 주파수 | 가속도 | 비고 |
|---|---|---|
| 5 – 100 Hz | 1.0 g | 페이로드 패어링 내부 기준 |

### 3.3 음향 환경 (Acoustic)

| 주파수 | OASPL | 비고 |
|---|---|---|
| 31.5 – 8000 Hz | 139 dB OASPL | 리프트오프 시 최대 |

---

## 4. 열 환경 (Thermal Environment)

| 구분 | 온도 | 비고 |
|---|---|---|
| 페이로드 베이 최저 | -20 °C | 궤도 진입 전 |
| 페이로드 베이 최고 | +50 °C | 이중화 열 처리 |
| 궤도 운용 최저 | -40 °C | 일면 냉복사 |
| 궤도 운용 최고 | +80 °C | 태양 직사 |

---

## 5. 부품 선정 기준 요약

Starlink Gen2 위성에 탑재되는 부품은 다음 기준을 충족해야 한다:

1. **TID 내성**: ≥ 15 krad(Si) (5년 LEO 마진 포함)
2. **SEL 면역**: LET > 60 MeV-cm²/mg 또는 SEL 면역 인증
3. **진동 내성**: ≥ 8.8 g rms (Falcon 9 PAF 기준, 3축)
4. **온도 범위**: -40 °C ~ +85 °C 이상
5. **준수 규격**: MIL-STD-883 Method 1019(방사선), MIL-STD-810H Method 514(진동)

---

## 6. 참고 규격 및 출처

- Falcon 9 Users Guide Rev 2.0 (SpaceX, 2021)
- Starlink Mission Overview (SpaceX, 2023)
- ECSS-E-ST-10-04C: Space Environment (ESA, 2008)
- NASA-HDBK-4002A: Mitigating In-Space Charging Effects
- MIL-STD-883 Method 1019: Ionizing Radiation (Total Dose) Test
- MIL-STD-810H Method 514.8: Vibration
