import json
import random

# ==============================================================================
# CONFIGURATION
# ==============================================================================
NUM_SAMPLES = 500  # How many sentences to generate
OUTPUT_DIR = "."

# ==============================================================================
# DATA POOLS
# ==============================================================================
GROUPS = [
    ("Lazarus", "Group"), ("APT28",), ("FIN7",), ("Wizard", "Spider"), 
    ("Sandworm", "Team"), ("Cozy", "Bear"), ("Deep", "Panda"), ("OilRig",),
    ("Silence", "Group"), ("Equation", "Group"), ("Carbanak",), ("MuddyWater",)
]

MALWARE = [
    ("Cobalt", "Strike"), ("TrickBot",), ("Emotet",), ("Ryuk",), ("Mimikatz",),
    ("BlackEnergy",), ("Zeus",), ("SpyEye",), ("WannaCry",), ("Petya",),
    ("Agent", "Tesla"), ("FormBook",), ("QakBot",)
]

TOOLS = [
    ("PowerShell",), ("PsExec",), ("Metasploit",), ("Nmap",), ("Wireshark",),
    ("Burp", "Suite"), ("BloodHound",), ("Empire",), ("net.exe",), ("cmd.exe",)
]

TACTICS = [
    ("Initial", "Access"), ("Execution",), ("Persistence",), ("Privilege", "Escalation"),
    ("Defense", "Evasion"), ("Credential", "Access"), ("Discovery",), 
    ("Lateral", "Movement"), ("Collection",), ("Exfiltration",), ("Command", "and", "Control")
]

TECHNIQUES = [
    ("Spear", "Phishing"), ("Drive-by", "Compromise"), ("PowerShell", "Execution"),
    ("Scheduled", "Task"), ("Registry", "Run", "Keys"), ("Process", "Injection"),
    ("Masquerading",), ("Credential", "Dumping"), ("Brute", "Force"), 
    ("T1059",), ("T1105",), ("T1003",)
]

DOMAINS = [
    ("evil.com",), ("bad-update.net",), ("phishing.org",), ("c2.server.io",),
    ("malicious-site.cn",), ("apt-infrastructure.ru",), ("update.windows-kernel.com",)
]

URLS = [
    ("http://bad.com/payload.exe",), ("https://site.org/login.php",), 
    ("http://192.168.1.50/setup.sh",), ("https://cdn.discord.com/malware.zip",)
]

CAMPAIGNS = [
    ("Operation", "Aurora"), ("SolarWinds", "Compromise"), ("Cloud", "Hopper"),
    ("Operation", "Dream", "Job"), ("Soft", "Cell")
]

TEMPLATES = [
    "The {GROUP} used {MALWARE} to target the network.",
    "Analysts observed {TACTIC} using {TOOL}.",
    "Traffic to {DOMAIN} indicates {MALWARE} infection.",
    "During {CAMPAIGN} , {GROUP} deployed {MALWARE} .",
    "{TECHNIQUE} was detected involving {TOOL} .",
    "Download the payload from {URL} .",
    "{GROUP} is known for {TACTIC} via {TECHNIQUE} .",
    "The file beaconed to {DOMAIN} using {TOOL} .",
    "IOCs include {URL} and {DOMAIN} .",
    "{MALWARE} executes {TECHNIQUE} for {TACTIC} ."
]

# ==============================================================================
# BIOUL LOGIC
# ==============================================================================
def get_bioul_tags(entity_tokens, label):
    if len(entity_tokens) == 1:
        return [f"U-{label}"]
    tags = [f"B-{label}"] + [f"I-{label}"] * (len(entity_tokens) - 2) + [f"L-{label}"]
    return tags

def generate_sentence():
    template = random.choice(TEMPLATES)
    
    replacements = {}
    
    # Select random entities
    group = random.choice(GROUPS)
    malware = random.choice(MALWARE)
    tool = random.choice(TOOLS)
    tactic = random.choice(TACTICS)
    technique = random.choice(TECHNIQUES)
    domain = random.choice(DOMAINS)
    url = random.choice(URLS)
    campaign = random.choice(CAMPAIGNS)

    # Tokenize sentence based on placeholders
    words = template.split()
    tokens = []
    ner_tags = []

    for word in words:
        if word == "{GROUP}":
            tokens.extend(group)
            ner_tags.extend(get_bioul_tags(group, "CTI_GROUP"))
        elif word == "{MALWARE}":
            tokens.extend(malware)
            ner_tags.extend(get_bioul_tags(malware, "MALWARE"))
        elif word == "{TOOL}":
            tokens.extend(tool)
            ner_tags.extend(get_bioul_tags(tool, "TOOL"))
        elif word == "{TACTIC}":
            tokens.extend(tactic)
            ner_tags.extend(get_bioul_tags(tactic, "MITRE_TACTIC"))
        elif word == "{TECHNIQUE}":
            tokens.extend(technique)
            ner_tags.extend(get_bioul_tags(technique, "MITRE_TECHNIQUE"))
        elif word == "{DOMAIN}":
            tokens.extend(domain)
            ner_tags.extend(get_bioul_tags(domain, "DOMAIN"))
        elif word == "{URL}":
            tokens.extend(url)
            ner_tags.extend(get_bioul_tags(url, "URL"))
        elif word == "{CAMPAIGN}":
            tokens.extend(campaign)
            ner_tags.extend(get_bioul_tags(campaign, "CTI_CAMPAIGN"))
        else:
            tokens.append(word)
            ner_tags.append("O")

    return {"tokens": tokens, "ner_tags": ner_tags}

# ==============================================================================
# EXECUTION
# ==============================================================================
data = [generate_sentence() for _ in range(NUM_SAMPLES)]

# Split 80/10/10
split_1 = int(NUM_SAMPLES * 0.8)
split_2 = int(NUM_SAMPLES * 0.9)

train = data[:split_1]
val = data[split_1:split_2]
test = data[split_2:]

with open("train.json", "w") as f: json.dump(train, f, indent=2)
with open("val.json", "w") as f: json.dump(val, f, indent=2)
with open("test.json", "w") as f: json.dump(test, f, indent=2)

print(f"Generated {len(train)} training, {len(val)} validation, and {len(test)} test samples.")