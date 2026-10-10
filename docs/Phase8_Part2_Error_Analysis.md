# Phase 8, Part 2: Error Analysis

## Kya Hai Error Analysis?

Sirf aggregate numbers (jaise "92 miss huay, 282 pakray gaye") dekhna kaafi nahi hota. Error Analysis mein hum un specific customers ko "alag nikal kar" dekhte hain jinhein model ne ghalat predict kiya, aur poochte hain: **"In sab mein koi common pattern hai?"**

Ye bilkul waisa hai jaise ek teacher apne un 92 students ko alag se dekhe jo fail honay walay thay lekin usne unhein "pass hoga" bola — aur poochay "kya ye sab kisi ek topic mein kamzor thay?"

## Do Tarah Ke "Miss" — Borderline vs Confident

- **Borderline Miss:** Model ne probability diya jo threshold (0.3) ke bohat qareeb tha (jaise 0.28) — sirf thori si kami rah gayi.
- **Confident Miss:** Model ne bohat kam probability diya (jaise 0.05) — yaani model ko "poora yaqeen" tha ye customer safe hai, lekin wo phir bhi chala gaya. Ye zyada khatarnak qisam ki galti hai.

## Humare Result (Test Set Se)

### False Negatives (92 Miss Huay Churners)
| Cheez | Value |
|---|---|
| Average tenure | 28.99 mahine |
| Average MonthlyCharges | $60.45 |
| Sabse common Contract | Month-to-month |
| Sabse common InternetService | **DSL** |
| Sabse common PaymentMethod | Mailed check |
| Median predicted probability | 0.1877 |
| Close calls (0.25–0.30) | 21 log (22.8%) |
| Confident misses (<0.25) | 71 log (**77.2%**) |

### False Positives (261 Fazool Alarm)
| Cheez | Value |
|---|---|
| Average tenure | 19.11 mahine |
| Average MonthlyCharges | $76.94 |
| Sabse common InternetService | Fiber optic |
| Sabse common PaymentMethod | Electronic check |

### True Positives (282 Sahi Pakray Gaye) — Comparison Ke Liye
| Cheez | Value |
|---|---|
| Average tenure | 12.38 mahine |
| Average MonthlyCharges | $76.78 |
| Sabse common InternetService | Fiber optic |

## Sabse Bari Discovery: Model Ka "Blind Spot"

Model ne do tarah ke churners mein se sirf ek tarah ko pehchana:

- **Type A — Model achi tarah pakarta hai:** Naya customer + Fiber optic + mehenga bill = "obvious khatra"
- **Type B — Model miss kar jata hai:** Purana customer + DSL + kam bill + Mailed check = "chhupa hua khatra"

**77.2% miss huay log "confident misses" thay, "close calls" nahi** — matlab ye koi statistical luck ka masla nahi, balke model ka ek genuine blind spot hai is customer segment ke liye.

## Zaroori Usool

Ye analysis sirf samajhne aur document karne ke liye hai. Model/threshold ko is se dobara tune nahi karenge, taake Test set ki "honesty" barqarar rahe.

## Future Improvement Ka Idea

Agle feature engineering mein zor un features par hona chahiye jo "subtle" churners (purane, saasta-bill, DSL walay) ko pakar sakein — na ke sirf "obvious" pattern par.
