# Mini-Quest Arena

An open-ended learning environment inspired by DeepMind's XLand research, designed for both human and AI agents to develop general problem-solving strategies through adaptive gameplay.

## 🎮 Game Concept

Mini-Quest Arena is a grid-based game where players (human or AI) must interpret text goals and plan actions accordingly. Each round features a randomly generated arena with unique objectives, promoting flexible thinking rather than memorization.

## 🚀 Features

### Core Gameplay
- **Procedurally Generated Arenas**: 10x10 grid worlds with walls, objects, and goal zones
- **Dynamic Goal System**: Text-based objectives that change each round
- **Multiple Object Types**: Colored cubes and spheres with different properties
- **Interactive Elements**: Pickup, drop, and interact actions
- **Scoring System**: Points for completion, efficiency bonuses, and penalties

### Game Modes
- **Single Player**: Classic individual gameplay
- **Cooperative Mode**: Two players work together to achieve shared goals
- **Competitive Mode**: Two players compete to outperform each other

### Goal Types
1. **Collection Goals**: "Collect the purple cube"
2. **Transport Goals**: "Bring the yellow sphere to the red goal zone"
3. **Avoidance Goals**: "Avoid the blue wall for 10 moves"
4. **Exploration Goals**: "Touch all corners before time runs out"
5. **Multi-Collection**: "Collect all red objects"
6. **Zone Clearing**: "Clear the green goal zone of all objects"
7. **Cooperative Goals**: "Work together to collect 5 objects"
8. **Competitive Goals**: "Collect more objects than your opponent!"

### AI Integration
- **AI Agent**: Toggle AI player for testing and demonstration
- **Adaptive Behavior**: AI attempts to interpret and complete goals
- **Learning Potential**: Framework suitable for reinforcement learning experiments

## 🎯 How to Play

### Controls
- **Arrow Keys / WASD**: Move player (up, down, left, right)
- **Space**: Pickup object
- **E**: Drop object
- **F**: Interact with environment
- **On-screen buttons**: Click to perform actions

### Game Mechanics
1. **Read the Goal**: Each round starts with a text objective
2. **Plan Your Strategy**: Analyze the arena and goal requirements
3. **Execute Actions**: Move, pickup, drop, and interact to complete objectives
4. **Score Points**: Earn points for completion, bonuses for efficiency
5. **Adapt to Change**: Each round generates a new arena and goal

### Multiplayer Controls
In multiplayer modes, players take turns or can use different control schemes:
- **Player 1**: Arrow keys
- **Player 2**: WASD keys (when implemented)

## 🧩 Technical Architecture

### Frontend Components
- **HTML5 Canvas**: Game rendering and visualization
- **Vanilla JavaScript**: Game engine and logic
- **CSS3**: Modern responsive UI design
- **No Dependencies**: Pure web technologies

### Game Engine Classes
```javascript
class MiniQuestArena {
    // Core game management
    - Arena generation
    - Player management
    - Goal assignment
    - Scoring system
    - Multi-agent coordination
}
```

### Key Systems
- **Grid World**: 2D array-based environment
- **Entity Management**: Players, objects, and zones
- **Goal Parser**: Text interpretation system
- **State Machine**: Game flow control
- **Event System**: User input and actions

## 🔧 Installation & Setup

1. **Clone or Download** the project files
2. **Open `index.html`** in a modern web browser
3. **No installation required** - runs entirely in the browser

### Browser Compatibility
- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## 🎨 Customization

### Modifying Game Parameters
```javascript
// In game.js
constructor() {
    this.gridSize = 10;        // Arena size
    this.cellSize = 60;        // Visual cell size
    this.maxInventorySize = 4; // Player inventory capacity
}
```

### Adding New Goals
```javascript
// Add to goals array in constructor()
{
    type: 'custom_goal',
    description: 'Custom goal description with {placeholders}'
}
```

### Creating New Objects
```javascript
// Extend colors object
colors: {
    customObject: { red: '#ff0000', blue: '#0000ff' }
}
```

## 🤖 AI Integration

The game includes a basic AI agent that can be toggled on/off. This provides a foundation for:
- **Reinforcement Learning**: State-action-reward cycles
- **Goal-Oriented Behavior**: Text-based objective parsing
- **Multi-Agent Training**: Competitive and cooperative scenarios
- **Zero-Shot Adaptation**: Handling novel goal configurations

### AI Enhancement Points
- Goal parsing and planning algorithms
- Pathfinding and navigation
- Object recognition and prioritization
- Strategic decision-making

## 📊 Research Applications

This implementation serves as a testbed for:
- **Open-Ended Learning**: Agents that adapt to novel tasks
- **General Intelligence**: Transfer across different goal types
- **Multi-Agent Systems**: Cooperation and competition
- **Human-AI Interaction**: Comparative performance studies
- **Curriculum Learning**: Progressive difficulty scaling

## 📚 Alignment with DeepMind's Open-Ended Play Research

Mini-Quest Arena mirrors the loop described in *“Open-Ended Learning Leads to Generally Capable Agents”* by combining:
- **Dynamic Task Generation** that keeps difficulty “just right,” powered by adaptive curricula and competency tracking.
- **Goal-Attentive (GOAT) Policies** that echo the blog’s attention-equipped recurrent agents for subgoal discovery.
- **Population-Based Training (PBT) & Generational Bootstrapping** so each cohort benefits from the previous one’s behaviors.
- **Behavior Analysis Pipeline** that surfaces experimentation, tool use, and cooperation heuristics seen in XLand probes.

## 🧪 Evaluation Methodology

To reflect the paper’s evaluation protocol, the ML stack now:
- Computes **Nash-style baselines** per held-out task and normalizes every agent’s score before comparison.
- Tracks **percentiles of normalized scores** (P10–P90) so an agent only “wins” if it dominates across the distribution.
- Reports **participation rate** — the percentage of tasks where an agent achieved non-zero reward, mirroring the blog’s participation metric.
- Bundles these metrics with classical zero-shot, few-shot, and transfer evaluations for a holistic generality score.

## 🧠 LLM-Assisted Control (Hybrid RL + Tiny LLM)

The `ml/llm_bridge.py` module lets a small instruction-tuned LLM read the textual goal, reason about the current world state, and issue actions through a **toolbox** that mirrors Mini-Quest Arena’s controls (`move`, `pickup`, `drop`, `interact`). This enables:
- Goal understanding + validation by the LLM before committing moves.
- Tool execution that feeds directly into the same environment surface the reinforcement learner uses.
- Hybrid agents where PPO provides fast reflexes while the LLM intervenes periodically for high-level decisions.

### Setup
1. Install the OpenAI-compatible client: `pip install openai`.
2. Copy `.env.example` to `.env` and insert your API credentials:
   ```bash
   cp .env.example .env
   # edit LLM_API_KEY, optional LLM_MODEL/BASE_URL overrides
   ```
3. (Optional) run the demo: `python ml_example.py --mode llm_demo --goal "Collect the purple cube"`

### Tiny-model API example
```python
from ml.llm_bridge import LLMGoalCoach, MiniQuestToolbox, SimpleMiniQuestController

controller = SimpleMiniQuestController()
toolbox = MiniQuestToolbox(controller)
coach = LLMGoalCoach(toolbox)

plan = coach.propose_plan(
    goal_text="Collect the purple cube and place it in the green zone.",
    state_summary=controller.summarize_state(),
    stream=True  # Streams tokens à la NVIDIA integrate endpoint
)

for action in plan.actions:
    result = toolbox.execute(action)
    print(f"Tool={action.tool} :: {result.observation}")
```
Behind the scenes this uses the OpenAI chat interface, so you can point it to NVIDIA’s `integrate.api.nvidia.com/v1` (or any compatible endpoint) via the `.env` file:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["LLM_API_KEY"],
)
completion = client.chat.completions.create(
    model="mistralai/mistral-7b-instruct-v0.3",
    messages=[{"role": "user", "content": "what is nuclear physics?"}],
    temperature=0.2,
    top_p=0.7,
    max_tokens=512,
    stream=True,
)
for chunk in completion:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```
Swap in task- and tool-specific prompts (as done in `LLMGoalCoach`) to let the model reason about goals, evaluate progress, and choose the next control to execute. The hybrid agent (`HybridLLMAgent`) can then alternate between RL actions and LLM-issued tool calls, giving you a playground for “LLM-improves-RL, RL-improves-LLM” experiments.

## 🎯 Future Enhancements

### Planned Features
- [ ] Advanced AI with learning algorithms
- [ ] More complex goal types and combinations
- [ ] Larger arena sizes and configurations
- [ ] Network multiplayer support
- [ ] Performance analytics and tracking
- [ ] Custom level editor
- [ ] Sound effects and audio feedback
- [ ] Mobile touch controls

### Research Extensions
- [ ] Integration with RL frameworks (OpenAI Gym, etc.)
- [ ] Neural network agent implementations
- [ ] Behavioral cloning from human gameplay
- [ ] Meta-learning across goal types
- [ ] Emergent behavior analysis

## 📄 License

This project is open-source and available for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Bug fixes and optimizations
- New goal types and game mechanics
- AI algorithm enhancements
- UI/UX improvements
- Documentation and examples

## 📞 Support

For questions, suggestions, or bug reports, please open an issue in the project repository.

---

**Built with inspiration from DeepMind's research on generally capable agents and open-ended learning.**
