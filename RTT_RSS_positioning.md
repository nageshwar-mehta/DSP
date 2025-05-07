Let me explain what each of the three positioning result figures shows and how to interpret them:

## 1. RTT Positioning Figure (Round-Trip Time)
**What it shows**:
- Estimates based solely on time-based distance measurements (how long signals take to travel)
- Typically more accurate than RSS in line-of-sight conditions
- Performance degrades with multipath interference (signals bouncing off surfaces)

**Key characteristics**:
- Best for environments with clear paths between devices and APs
- More consistent accuracy than RSS (smaller variance in errors)
- Errors typically appear as small random offsets from ground truth
- Performance drops significantly when walls/obstacles block direct paths

**What to look for**:
- Many short error lines (blue dashes) indicate good accuracy
- Clusters of large errors may reveal problematic areas
- Symmetric error distribution suggests measurement noise
- Biased errors (all in one direction) suggest systematic issues

## 2. RSS Positioning Figure (Received Signal Strength)
**What it shows**:
- Estimates based on signal power measurements
- More affected by environmental factors than RTT
- Typically less accurate but works in non-line-of-sight conditions

**Key characteristics**:
- Larger errors than RTT in most cases
- Errors often show patterns (e.g., consistent under/over-estimation)
- Performance depends heavily on proper path loss modeling
- More sensitive to device orientation and antenna characteristics

**What to look for**:
- Longer blue dashes overall compared to RTT plot
- Error directions may reveal signal blockage patterns
- Consistent bias indicates calibration needed (e.g., path loss exponent)
- Large outliers suggest multipath or interference issues

## 3. RTT+RSS Combined Positioning Figure
**What it shows**:
- Fusion of both measurement types
- Aims to combine RTT's precision with RSS's robustness
- Should provide the most reliable overall performance

**Key characteristics**:
- Typically shows intermediate error magnitude between RTT and RSS
- More consistent than RSS alone
- More robust than RTT alone in challenging environments
- Errors should be smaller and more evenly distributed

**What to look for**:
- Should have fewer extreme outliers than either method alone
- Error lines should be shorter than RSS-only, though maybe slightly longer than RTT-only
- Combination should compensate for each method's weaknesses
- Check if mean error is lower than either individual method

## Comparative Analysis Table

| Aspect        | RTT Positioning          | RSS Positioning          | Combined RTT+RSS         |
|---------------|-------------------------|-------------------------|--------------------------|
| **Accuracy**  | High in LOS conditions   | Moderate                | Balanced                 |
| **Precision** | Consistent              | Variable                | Improved consistency     |
| **Robustness**| Weak to obstructions    | Handles obstructions    | Most robust              |
| **Error Pattern** | Small random offsets | Larger, possibly biased | Balanced, reduced bias   |
| **Best For**  | Open spaces             | Complex environments    | General-purpose use      |
| **Typical Use Case** | High-accuracy needs | Backup when RTT fails   | Reliable everyday use    |

The visualization lets you verify these theoretical expectations with your actual deployment data. 
