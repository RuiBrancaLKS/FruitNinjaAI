# Fruit Ninja AI Bot

This repository contains an AI bot designed to play the game Fruit Ninja automatically. The bot uses a fine-tuned YOLO11n model to detect fruits and bombs, allowing it to interact with the game and avoid hitting bombs. The AI bot is capable of playing at three different resolutions (256, 320, and 640), with accuracy increasing as resolution increases, while speed decreases. The bot is compatible with BlueStacks to capture the game screen and perform actions accordingly.

## Table of Contents

- [Models Available](#models-available)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [How to Use](#how-to-use)
- [License](#license)

## Models Available

The AI bot offers three models that balance speed and accuracy:

- **256**: Fastest, lowest accuracy.
- **320**: Moderate speed and accuracy.
- **640**: Highest accuracy but slower.

These models are fine-tuned on custom datasets for the Fruit Ninja game. Choose the resolution that best fits your hardware capabilities and performance needs.

## Prerequisites

Ensure you have the following installed on your system:

- Python 3.x
- PyTorch
- BlueStacks (to run Fruit Ninja)
- Additional Python libraries (see ```bash requirements.txt``` for details)

- [Cuda](https://pytorch.org/get-started/locally/) (optional)

## Installation

1. **Clone the repository:**
``` bash
 git clone https://github.com/yourusername/fruit-ninja-ai-bot.git
 cd fruit-ninja-ai-bot
```

2. **Install the required Python libraries:**
``` bash
pip install -r requirements.txt
```

3. **[(optional) Install pytorch with cuda 12.4](https://pytorch.org/get-started/locally/)**

4. **Configure BlueStacks to run Fruit Ninja:**

## How to Use

Run the bot with the following command:

```bash
python run.py <resolution> <training_flag> <debug_flag>
```

- **resolution**: Choose between ```256```, ```300```, or ```640``` (default is 320 if an invalid resolution is given).

- **training_flag**: Use ```-T``` or ```-Train``` to train the model further with your custom dataset.

- **debug_flag**: User ```-d``` or ```-debug``` for debug mode, where screenshots and predictions are saved for each action.

##### Example

``` bash
python run.py 320 -d
```

This will run the bot using the 320x320 model in debug mode.

## Training the AI

To train the AI on a new dataset, run:

```python run.py 320 -T```

Ensure your dataset is configured correctly under ```datasets/{resolution}x{resolution}/data.yaml```

## License

This project includes two different licenses:

1. My Code (everything except the YOLO model): Licensed under the MIT License.
    - You are free to use, modify, and distribute the code in any way, as long as you include the original copyright notice and this license in any copies or substantial portions of the software.

    **MIT License:**
    ```
    Copyright 2024 RuiBranca

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    ```

2. **YOLO Model:** Licensed under the AGPL-3.0 by Ultralytics.
    - The YOLO model is provided under the AGPL-3.0 license. If you modify or integrate the model in a system that is accessible over a network (e.g., a web service), you must share the source code of your entire project under the AGPL-3.0 license.
    - For more details, see the AGPL-3.0 License and the Ultralytics License.
    - You can see more at [ultralytics github page](https://github.com/ultralytics/ultralytics).
 