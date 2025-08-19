# 🐿️ Blind Squirrel Agent

#### 🥈 2nd Place Winner - [ARC-AGI-3 Preview Competition](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings)

#### 🔧 Developed by Will Dick


## How to Run

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if not aready installed.
   
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone this repo branch.
   
   ```bash
   git clone --branch blindsquirrel https://github.com/wd13ca/ARC-AGI-3-Agents.git
   ```

3. Enter the directory.
   
   ```bash
   cd ARC-AGI-3-Agents
   ```

3. Copy .env.example to .env
   
   ```bash
   cp .env.example .env
   ```

4. Get an API key from the [ARC-AGI-3 Website](https://three.arcprize.org/) and set it as an environment variable in your .env file.
   
   ```bash
   export ARC_API_KEY="your_api_key_here"
   ```

5. Run the **Blind Squirrel** agent.
   
   ```bash
   uv run main.py --agent=blindsquirrel
   ```
