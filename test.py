# save_and_run.py
import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()
random.seed(42)
Faker.seed(42)

# ---------- configuration ----------
NUM_ROWS = 1000
TECHNICIANS = ["Elena Vasquez", "Marcus Chen", "Sophia Patel", "James O'Brien",
               "Fatima Al-Rashid", "David Kim", "Linda Schmidt", "Carlos Mendez"]
END_DATE = datetime.now().date()
START_DATE = END_DATE - timedelta(days=3*365)

# ---------- domain logic ----------
MACHINE_FAULT_RESOLUTION = {
    "DC Electric Motor": {
        "faults": ["Overheating under load", "Rotor jammed", "Excessive sparking at brushes",
                   "Intermittent operation", "Loss of torque", "Abnormal vibration"],
        "resolutions": [
            "Replaced carbon commutator brushes and cleaned commutator. Realigned poles using Fleming's Left Hand Rule.",
            "Applied Fleming's Left Hand Rule to verify field orientation. Replaced worn carbon commutator brushes.",
            "Overhauled motor: replaced carbon commutator brushes, inspected armature, used Fleming's Left Hand Rule.",
            "Disassembled rotor, removed debris, replaced carbon commutator brushes. Realigned stator poles per Fleming's Left Hand Rule.",
            "Measured field coil resistance, replaced defective winding, installed new carbon commutator brushes. Verified operation using Fleming's Left Hand Rule."
        ]
    },
    "Solenoid Coil": {
        "faults": ["Coil open circuit", "Coil shorted turns", "Plunger stuck in housing",
                   "Excessive heat generation", "Intermittent actuation", "Humming noise without movement"],
        "resolutions": [
            "Replaced solenoid coil assembly. Measured resistance 24 ohms, verified plunger movement.",
            "Removed burnt coil, cleaned core, installed new solenoid coil. Checked duty cycle.",
            "Disassembled solenoid, cleaned plunger, lubricated. Replaced coil due to partial short.",
            "Tested coil inductance, found shorted turns. Replaced with OEM solenoid coil.",
            "Inspected supply voltage, found intermittent connection. Replaced solenoid coil and tightened terminals."
        ]
    },
    "Magnetic Compass Assembly": {
        "faults": ["Compass needle sticking", "Inaccurate bearing reading", "Liquid leakage",
                   "Card tilted or sluggish", "Night lighting failure", "Lubber line misaligned"],
        "resolutions": [
            "Replaced magnetic compass assembly. Demagnetized housing, refilled with damping fluid.",
            "Drained and refilled compass fluid, replaced needle bearing. Recalibrated using known headings.",
            "Sealed leak with epoxy, refilled with isopar L fluid. Checked card float and friction.",
            "Removed corrosion from pivot, replaced jewel bearing. Verified accuracy against GPS.",
            "Installed new LED lighting module. Realigned lubber line with vessel centerline."
        ]
    },
    "Testing Rig": {
        "faults": ["Galvanometer shows erratic deflection", "Galvanometer needle stuck at zero",
                   "Galvanometer calibration drift", "Galvanometer overload at low current",
                   "Galvanometer damping too high"],
        "resolutions": [
            "Replaced galvanometer with calibrated unit. Verified shunt resistors.",
            "Adjusted galvanometer zero screw and cleaned jewel bearings. Applied known current.",
            "Recalibrated galvanometer using precision resistor decade box. Adjusted spring tension.",
            "Installed new galvanometer movement. Checked overload protection circuit.",
            "Reduced damping fluid viscosity and recalibrated galvanometer scale."
        ]
    }
}
OTHER_MACHINES = ["Induction Motor", "Servo Actuator", "Transformer", "Pneumatic Valve"]

# ---------- generators ----------
def generate_ticket_id(i): return f"TKT-{1000+i}"
def generate_date(): return (START_DATE + timedelta(days=random.randint(0, (END_DATE-START_DATE).days))).strftime("%Y-%m-%d")
def generate_downtime_hours(): return round(random.uniform(0.5, 14.0), 1)
def generate_machine_type(): return random.choice(list(MACHINE_FAULT_RESOLUTION.keys())) if random.random() < 0.7 else random.choice(OTHER_MACHINES)

def generate_fault_and_resolution(machine_type):
    if machine_type in MACHINE_FAULT_RESOLUTION:
        data = MACHINE_FAULT_RESOLUTION[machine_type]
        fault = random.choice(data["faults"])
        resolution = random.choice(data["resolutions"])
    else:
        fault = random.choice(["Unusual noise", "Intermittent power", "Overheating", "Unexpected shutdown", "Reduced efficiency"])
        resolution = random.choice(["Inspected connections, replaced relay.", "Cleaned cooling fins, verified airflow.",
                                    "Ran diagnostic, recalibrated control board.", "Replaced cable harness, tested load.",
                                    "Updated firmware, performed functional test."])
        if random.random() < 0.3:
            hooks = ["Carbon Commutator Brushes", "Fleming's Left Hand Rule", "Galvanometer", "Solenoid Coil", "Magnetic Compass Assembly", "DC Electric Motor"]
            resolution += f" Also noted that {random.choice(hooks)} was within specifications."
    if machine_type == "DC Electric Motor":
        if "Carbon Commutator Brushes" not in resolution and "Fleming's Left Hand Rule" not in resolution:
            resolution += " Performed pole alignment using Fleming's Left Hand Rule and replaced carbon commutator brushes."
    if machine_type == "Testing Rig" and "Galvanometer" not in fault:
        fault = "Galvanometer " + fault.lower()
    return fault, resolution

# ---------- generate dataframe ----------
records = []
for i in range(NUM_ROWS):
    mt = generate_machine_type()
    fault, resolution = generate_fault_and_resolution(mt)
    records.append({
        "Ticket_ID": generate_ticket_id(i),
        "Date": generate_date(),
        "Machine_Type": mt,
        "Reported_Fault": fault,
        "Technician": random.choice(TECHNICIANS),
        "Downtime_Hours": generate_downtime_hours(),
        "Resolution_Notes": resolution
    })

df = pd.DataFrame(records)
df.to_csv("maintenance_history_1000.csv", index=False)
print("CSV generated: maintenance_history_1000.csv")