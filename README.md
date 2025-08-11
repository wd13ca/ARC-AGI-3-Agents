# ARC-AGI-3 Preview Competition Entry

This is an entry to the [ARC-AGI-3 Preview Competition](https://arcprize.org/competitions/arc-agi-3-preview-agents/). It is a branch of the [ARC-AGI-3-Agents repo](https://github.com/arcprize/ARC-AGI-3-Agents). 

**Leadboard Name**  
wd13  

**Agent Name**  
Blind Squirrel  

**Developer**  
Will Dick  


## How to Run

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if not aready installed.
   
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone the this repo branch.
   
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
