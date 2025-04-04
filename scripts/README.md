# Würstchen diffuser uses separate environment (for background generation)
Find below instructions for creating synthetic backgrounds using the Würstchen model.

## Setup

Python 3.9.17.

CUDA 1.17

### Venv

Create.
```
python -m venv dataset_diffuser
```

Activate
```
source ./dataset_diffuser/bin/activate
```

Install requirements (exact requirements from `pip freeze`)
```
pip -r background_randomization/diffuser_exact_requirements.txt
```

Use Würstchen
See: [generate_backgrounds.py](./generate_backgrounds.py)
```
from background_randomization import WuerstchenBackgroundGenerator

background_types = ["cloudscape", "space", "jungle", "desert", "arctic", "volcanic", "ocean", "abstract_patterns"]

prompts = [
    f"A vast sky filled with dramatic clouds in varying shapes and layers, emphasizing realistic lighting and textures, devoid of objects or creatures."
    if background == "cloudscape" else
    f"A vast space skybox with stars, galaxies, and nebulae, creating a serene cosmic atmosphere, devoid of objects or creatures."
    if background == "space" else
    f"A tranquil underwater ocean scene with soft light beams filtering through the water, focusing on the textures of sand, water ripples, and vibrant coral formations."
    if background == "ocean" else
    f"A volcanic scene with a dark sky, an erupting volcano, glowing red lava flows, and ash clouds in the atmosphere."
    if background == "volcanic" else
    f"A {background} background scene with a realistic and immersive environment, devoid of any objects, creatures, or vehicles."
    if "abstract" not in background else
    f"An {background.replace('_', ' ')} background with smooth, surreal elements and vibrant colors."
    for background in background_types
]

generator = WuerstchenBackgroundGenerator()
generator.generate_backgrounds(prompts, background_types, n=100)
```