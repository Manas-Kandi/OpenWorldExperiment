const DEFAULT_LLM_API_BASE_URL = window.LLM_API_BASE_URL || 'http://localhost:8001';

class MiniQuestArena {
    constructor() {
        this.canvas = document.getElementById('game-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.gridSize = 10;
        this.cellSize = 60;
        this.grid = [];
        this.player = null;
        this.players = [];
        this.currentPlayerIndex = 0;
        this.gameMode = 'single'; // 'single', 'cooperative', 'competitive'
        this.objects = [];
        this.goalZones = [];
        this.currentGoal = null;
        this.score = 0;
        this.scores = {};
        this.moves = 0;
        this.playerMoves = {};
        this.startTime = null;
        this.gameTime = 0;
        this.aiEnabled = false;
        this.aiPlayers = [];
        this.inventory = [];
        this.maxInventorySize = 4;
        this.roundNumber = 1;
        this.timerInterval = null;
        this.llmActionQueue = [];
        this.llmProcessing = false;
        this.llmApiBaseUrl = DEFAULT_LLM_API_BASE_URL;
        this.llmThought = '';
        
        this.colors = {
            player: ['#10b981', '#f59e0b', '#ef4444', '#3b82f6'],
            wall: '#4b5563',
            empty: '#1a1a1a',
            cube: { red: '#ef4444', blue: '#3b82f6', green: '#10b981', yellow: '#f59e0b', purple: '#8b5cf6' },
            sphere: { red: '#ef4444', blue: '#3b82f6', green: '#10b981', yellow: '#f59e0b', purple: '#8b5cf6' },
            goalZone: { red: '#ef4444', blue: '#3b82f6', green: '#10b981', yellow: '#f59e0b', purple: '#8b5cf6' }
        };
        
        this.goals = [
            { type: 'collect', description: 'Collect the {color} {shape}.' },
            { type: 'bring_to_zone', description: 'Bring the {color} {shape} to the {zoneColor} goal zone.' },
            { type: 'avoid_walls', description: 'Avoid the {color} wall for {moves} moves.' },
            { type: 'touch_corners', description: 'Touch all corners before time runs out.' },
            { type: 'collect_multiple', description: 'Collect all {color} objects.' },
            { type: 'clear_zone', description: 'Clear the {color} goal zone of all objects.' },
            { type: 'pattern_match', description: 'Create a pattern: {pattern}' },
            { type: 'cooperative_collect', description: 'Work together to collect {count} objects.' },
            { type: 'competitive_collect', description: 'Collect more objects than your opponent!' },
            { type: 'cooperative_zone', description: 'Both players reach the {color} goal zone.' },
            { type: 'competitive_race', description: 'Be the first to touch all corners!' }
        ];
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.startNewGame();
    }
    
    setupEventListeners() {
        // Movement controls
        document.querySelectorAll('[data-action^="move-"]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const direction = e.target.closest('button').dataset.action.split('-')[1];
                this.movePlayer(direction);
            });
        });
        
        // Action controls
        document.querySelectorAll('[data-action="pickup"]').forEach(btn => {
            btn.addEventListener('click', () => this.pickupObject());
        });
        
        document.querySelectorAll('[data-action="drop"]').forEach(btn => {
            btn.addEventListener('click', () => this.dropObject());
        });
        
        document.querySelectorAll('[data-action="interact"]').forEach(btn => {
            btn.addEventListener('click', () => this.interact());
        });
        
        // Game controls
        document.getElementById('new-game-btn').addEventListener('click', () => this.startNewRound());
        document.getElementById('reset-btn').addEventListener('click', () => this.resetGame());
        document.getElementById('ai-toggle-btn').addEventListener('click', () => this.toggleAI());
        
        // Add game mode selection
        this.addGameModeControls();
        
        // Keyboard controls
        document.addEventListener('keydown', (e) => {
            if (this.aiEnabled) return;
            
            switch(e.key) {
                case 'ArrowUp':
                case 'w':
                    this.movePlayer('up');
                    break;
                case 'ArrowDown':
                case 's':
                    this.movePlayer('down');
                    break;
                case 'ArrowLeft':
                case 'a':
                    this.movePlayer('left');
                    break;
                case 'ArrowRight':
                case 'd':
                    this.movePlayer('right');
                    break;
                case ' ':
                    this.pickupObject();
                    break;
                case 'e':
                    this.dropObject();
                    break;
                case 'f':
                    this.interact();
                    break;
            }
        });
    }
    
    startNewGame() {
        this.score = 0;
        this.scores = {};
        this.roundNumber = 1;
        this.updateUI();
        this.startNewRound();
    }
    
    addGameModeControls() {
        const gameControls = document.querySelector('.game-controls');
        const modeSelector = document.createElement('div');
        modeSelector.className = 'mode-selector';
        modeSelector.innerHTML = `
            <select id="game-mode-select" class="mode-select">
                <option value="single">Single Player</option>
                <option value="cooperative">Cooperative (2 Players)</option>
                <option value="competitive">Competitive (2 Players)</option>
            </select>
        `;
        gameControls.insertBefore(modeSelector, gameControls.firstChild);
        
        document.getElementById('game-mode-select').addEventListener('change', (e) => {
            this.gameMode = e.target.value;
            this.resetGame();
        });
    }
    
    spawnPlayer() {
        this.players = [];
        this.scores = {};
        this.playerMoves = {};
        
        if (this.gameMode === 'single') {
            let x, y;
            do {
                x = Math.floor(Math.random() * this.gridSize);
                y = Math.floor(Math.random() * this.gridSize);
            } while (this.grid[y][x] !== 'empty' || this.isGoalZone(x, y) || this.getObjectAt(x, y));
            
            this.player = { x, y, inventory: [], id: 'player1', color: this.colors.player[0] };
            this.players = [this.player];
            this.scores['player1'] = 0;
            this.playerMoves['player1'] = 0;
        } else {
            // Spawn two players for multiplayer modes
            for (let i = 0; i < 2; i++) {
                let x, y;
                do {
                    x = Math.floor(Math.random() * this.gridSize);
                    y = Math.floor(Math.random() * this.gridSize);
                } while (this.grid[y][x] !== 'empty' || this.isGoalZone(x, y) || 
                        this.getObjectAt(x, y) || this.players.some(p => p.x === x && p.y === y));
                
                const player = { x, y, inventory: [], id: `player${i + 1}`, color: this.colors.player[i] };
                this.players.push(player);
                this.scores[`player${i + 1}`] = 0;
                this.playerMoves[`player${i + 1}`] = 0;
            }
            
            this.player = this.players[0]; // Set first player as active for controls
        }
        
        this.inventory = this.player.inventory;
    }
    
    startNewRound() {
        this.moves = 0;
        this.inventory = [];
        this.resetLLMState();
        this.generateArena();
        this.assignGoal();
        this.spawnPlayer();
        this.startTime = Date.now();
        this.hideMessage();
        this.hideGameOverModal();
        this.startTimer();
        this.updateUI();
        this.render();
        
        if (this.aiEnabled) {
            this.resetLLMState(true);
            this.processLLMLoop();
        }
    }
    
    generateArena() {
        // Initialize empty grid
        this.grid = Array(this.gridSize).fill().map(() => Array(this.gridSize).fill('empty'));
        this.objects = [];
        this.goalZones = [];
        
        // Add walls
        const wallCount = Math.floor(Math.random() * 8) + 5;
        for (let i = 0; i < wallCount; i++) {
            let x, y;
            do {
                x = Math.floor(Math.random() * this.gridSize);
                y = Math.floor(Math.random() * this.gridSize);
            } while (this.grid[y][x] !== 'empty');
            this.grid[y][x] = 'wall';
        }
        
        // Add goal zones
        const zoneCount = Math.floor(Math.random() * 3) + 1;
        const zoneColors = ['red', 'blue', 'green', 'yellow', 'purple'];
        for (let i = 0; i < zoneCount; i++) {
            let x, y;
            do {
                x = Math.floor(Math.random() * this.gridSize);
                y = Math.floor(Math.random() * this.gridSize);
            } while (this.grid[y][x] !== 'empty');
            
            const color = zoneColors[Math.floor(Math.random() * zoneColors.length)];
            this.goalZones.push({ x, y, color, type: 'zone' });
        }
        
        // Add objects
        const objectCount = Math.floor(Math.random() * 8) + 5;
        const shapes = ['cube', 'sphere'];
        const colors = ['red', 'blue', 'green', 'yellow', 'purple'];
        
        for (let i = 0; i < objectCount; i++) {
            let x, y;
            do {
                x = Math.floor(Math.random() * this.gridSize);
                y = Math.floor(Math.random() * this.gridSize);
            } while (this.grid[y][x] !== 'empty' || this.isGoalZone(x, y));
            
            const shape = shapes[Math.floor(Math.random() * shapes.length)];
            const color = colors[Math.floor(Math.random() * colors.length)];
            this.objects.push({ x, y, shape, color, id: `obj_${i}` });
        }
    }
    
    isGoalZone(x, y) {
        return this.goalZones.some(zone => zone.x === x && zone.y === y);
    }
    
    assignGoal() {
        let goalTemplate;
        
        if (this.gameMode === 'cooperative') {
            const cooperativeGoals = this.goals.filter(g => g.type.includes('cooperative'));
            goalTemplate = cooperativeGoals[Math.floor(Math.random() * cooperativeGoals.length)];
        } else if (this.gameMode === 'competitive') {
            const competitiveGoals = this.goals.filter(g => g.type.includes('competitive'));
            goalTemplate = competitiveGoals[Math.floor(Math.random() * competitiveGoals.length)];
        } else {
            const singlePlayerGoals = this.goals.filter(g => !g.type.includes('cooperative') && !g.type.includes('competitive'));
            goalTemplate = singlePlayerGoals[Math.floor(Math.random() * singlePlayerGoals.length)];
        }
        
        const colors = ['red', 'blue', 'green', 'yellow', 'purple'];
        const shapes = ['cube', 'sphere'];
        
        let description = goalTemplate.description;
        let goalData = { ...goalTemplate };
        
        // Replace placeholders
        description = description.replace('{color}', colors[Math.floor(Math.random() * colors.length)]);
        description = description.replace('{shape}', shapes[Math.floor(Math.random() * shapes.length)]);
        description = description.replace('{zoneColor}', colors[Math.floor(Math.random() * colors.length)]);
        description = description.replace('{moves}', Math.floor(Math.random() * 10) + 5);
        description = description.replace('{count}', Math.floor(Math.random() * 5) + 3);
        description = description.replace('{pattern}', 'red-blue-green');
        
        this.currentGoal = {
            ...goalData,
            description,
            completed: false,
            progress: 0,
            targetColor: description.match(/(\w+)\s+(cube|sphere)/)?.[1] || 'red',
            targetShape: description.match(/cube|sphere/)?.[0] || 'cube',
            targetZoneColor: description.match(/(\w+)\s+goal\s+zone/)?.[1] || 'red',
            targetMoves: parseInt(description.match(/(\d+)\s+moves/)?.[1] || '10'),
            targetCount: parseInt(description.match(/(\d+)\s+objects/)?.[1] || '3')
        };
        
        document.getElementById('goal-text').textContent = description;
    }
    
    movePlayer(direction) {
        if (!this.player || this.currentGoal?.completed) return;
        
        let newX = this.player.x;
        let newY = this.player.y;
        
        switch(direction) {
            case 'up': newY--; break;
            case 'down': newY++; break;
            case 'left': newX--; break;
            case 'right': newX++; break;
        }
        
        // Check boundaries
        if (newX < 0 || newX >= this.gridSize || newY < 0 || newY >= this.gridSize) {
            return;
        }
        
        // Check walls
        if (this.grid[newY][newX] === 'wall') {
            this.checkAvoidWallsGoal();
            return;
        }
        
        this.player.x = newX;
        this.player.y = newY;
        this.moves++;
        
        this.checkGoalProgress();
        this.updateUI();
        this.render();
    }
    
    pickupObject() {
        if (!this.player || this.inventory.length >= this.maxInventorySize) return;
        
        const object = this.getObjectAt(this.player.x, this.player.y);
        if (object) {
            this.inventory.push(object);
            this.objects = this.objects.filter(obj => obj.id !== object.id);
            this.updateInventoryUI();
            this.checkGoalProgress();
            this.render();
        }
    }
    
    dropObject() {
        if (!this.player || this.inventory.length === 0) return;
        
        const object = this.inventory[this.inventory.length - 1];
        if (!this.getObjectAt(this.player.x, this.player.y)) {
            object.x = this.player.x;
            object.y = this.player.y;
            this.objects.push(object);
            this.inventory.pop();
            this.updateInventoryUI();
            this.checkGoalProgress();
            this.render();
        }
    }
    
    interact() {
        if (!this.player) return;
        
        // Check if player is on a goal zone
        const zone = this.goalZones.find(z => z.x === this.player.x && z.y === this.player.y);
        if (zone) {
            this.checkGoalProgress();
        }
    }
    
    getObjectAt(x, y) {
        return this.objects.find(obj => obj.x === x && obj.y === y);
    }
    
    checkGoalProgress() {
        if (!this.currentGoal || this.currentGoal.completed) return;
        
        switch(this.currentGoal.type) {
            case 'collect':
                this.checkCollectGoal();
                break;
            case 'bring_to_zone':
                this.checkBringToZoneGoal();
                break;
            case 'avoid_walls':
                // Checked in movePlayer
                break;
            case 'touch_corners':
                this.checkTouchCornersGoal();
                break;
            case 'collect_multiple':
                this.checkCollectMultipleGoal();
                break;
            case 'clear_zone':
                this.checkClearZoneGoal();
                break;
            case 'cooperative_collect':
                this.checkCooperativeCollectGoal();
                break;
            case 'competitive_collect':
                this.checkCompetitiveCollectGoal();
                break;
            case 'cooperative_zone':
                this.checkCooperativeZoneGoal();
                break;
            case 'competitive_race':
                this.checkCompetitiveRaceGoal();
                break;
        }
    }
    
    checkCooperativeCollectGoal() {
        const totalCollected = this.players.reduce((total, player) => {
            return total + player.inventory.filter(obj => obj.color === this.currentGoal.targetColor).length;
        }, 0);
        
        if (totalCollected >= this.currentGoal.targetCount) {
            this.completeGoal(150);
        }
    }
    
    checkCompetitiveCollectGoal() {
        const player1Count = this.players[0].inventory.length;
        const player2Count = this.players[1].inventory.length;
        
        // Check if one player has significantly more objects
        if (player1Count >= 5 || player2Count >= 5) {
            const winner = player1Count > player2Count ? 'Player 1' : 'Player 2';
            this.currentGoal.completed = true;
            this.showMessage('Competitive Goal Complete!', `${winner} wins with ${Math.max(player1Count, player2Count)} objects!`, 'success');
            
            // Award points to winner
            if (winner === 'Player 1') {
                this.scores['player1'] += 100;
            } else {
                this.scores['player2'] += 100;
            }
            this.updateUI();
            setTimeout(() => this.showGameOverModal(), 1500);
        }
    }
    
    checkCooperativeZoneGoal() {
        const allPlayersOnZone = this.players.every(player => {
            return this.goalZones.some(zone => 
                zone.x === player.x && 
                zone.y === player.y && 
                zone.color === this.currentGoal.targetZoneColor
            );
        });
        
        if (allPlayersOnZone) {
            this.completeGoal(120);
        }
    }
    
    checkCompetitiveRaceGoal() {
        const corners = [
            {x: 0, y: 0}, {x: this.gridSize-1, y: 0},
            {x: 0, y: this.gridSize-1}, {x: this.gridSize-1, y: this.gridSize-1}
        ];
        
        for (let i = 0; i < this.players.length; i++) {
            const player = this.players[i];
            const touchedCorners = corners.filter(corner => 
                player.x === corner.x && player.y === corner.y
            ).length;
            
            if (touchedCorners === 4) {
                this.currentGoal.completed = true;
                this.showMessage('Race Won!', `Player ${i + 1} touched all corners first!`, 'success');
                this.scores[`player${i + 1}`] += 150;
                this.updateUI();
                setTimeout(() => this.showGameOverModal(), 1500);
                break;
            }
        }
    }
    
    checkCollectGoal() {
        const hasObject = this.inventory.some(obj => 
            obj.color === this.currentGoal.targetColor && 
            obj.shape === this.currentGoal.targetShape
        );
        
        if (hasObject) {
            this.completeGoal(100);
        }
    }
    
    checkBringToZoneGoal() {
        const hasObject = this.inventory.some(obj => 
            obj.color === this.currentGoal.targetColor && 
            obj.shape === this.currentGoal.targetShape
        );
        
        const onCorrectZone = this.goalZones.some(zone => 
            zone.x === this.player.x && 
            zone.y === this.player.y && 
            zone.color === this.currentGoal.targetZoneColor
        );
        
        if (hasObject && onCorrectZone) {
            this.completeGoal(150);
        }
    }
    
    checkAvoidWallsGoal() {
        // Wall collision detected - fail goal
        this.currentGoal.completed = true;
        this.showMessage('Goal Failed!', 'You touched a wall! -50 points', 'danger');
        this.score = Math.max(0, this.score - 50);
        this.updateUI();
        setTimeout(() => this.startNewRound(), 2000);
    }
    
    checkTouchCornersGoal() {
        const corners = [
            {x: 0, y: 0}, {x: this.gridSize-1, y: 0},
            {x: 0, y: this.gridSize-1}, {x: this.gridSize-1, y: this.gridSize-1}
        ];
        
        const touchedCorners = corners.filter(corner => 
            this.player.x === corner.x && this.player.y === corner.y
        ).length;
        
        if (touchedCorners === 4) {
            this.completeGoal(200);
        }
    }
    
    checkCollectMultipleGoal() {
        const targetObjects = this.inventory.filter(obj => 
            obj.color === this.currentGoal.targetColor
        );
        
        if (targetObjects.length >= 3) {
            this.completeGoal(120);
        }
    }
    
    checkClearZoneGoal() {
        const zone = this.goalZones.find(zone => 
            zone.color === this.currentGoal.targetZoneColor
        );
        
        const objectsInZone = this.objects.filter(obj => 
            obj.x === zone.x && obj.y === zone.y
        );
        
        if (objectsInZone.length === 0) {
            this.completeGoal(80);
        }
    }
    
    completeGoal(points) {
        this.currentGoal.completed = true;
        
        // Calculate efficiency bonus
        const efficiencyBonus = Math.max(0, 50 - this.moves);
        const totalPoints = points + efficiencyBonus;
        
        this.score += totalPoints;
        this.showMessage('Goal Complete!', `+${points} points +${efficiencyBonus} efficiency bonus`, 'success');
        this.updateUI();
        
        setTimeout(() => this.showGameOverModal(), 1500);
    }
    
    startTimer() {
        if (this.timerInterval) clearInterval(this.timerInterval);
        
        this.timerInterval = setInterval(() => {
            this.gameTime = Math.floor((Date.now() - this.startTime) / 1000);
            this.updateTimerUI();
            
            // Update goal timer bar
            const timeLimit = 120; // 2 minutes per round
            const progress = Math.max(0, 100 - (this.gameTime / timeLimit * 100));
            document.querySelector('.timer-bar').style.width = progress + '%';
            
            if (this.gameTime >= timeLimit) {
                this.timeUp();
            }
        }, 100);
    }
    
    timeUp() {
        clearInterval(this.timerInterval);
        this.showMessage('Time\'s Up!', 'Round failed due to time limit.', 'danger');
        this.score = Math.max(0, this.score - 25);
        this.updateUI();
        setTimeout(() => this.startNewRound(), 2000);
    }
    
    toggleAI() {
        this.aiEnabled = !this.aiEnabled;
        this.updateAIToggleUI();
        
        if (this.aiEnabled) {
            this.resetLLMState(true);
            this.updateLLMStatus('thinking');
            this.processLLMLoop();
        } else {
            this.stopLLMControl('LLM paused.');
        }
    }
    
    updateAIToggleUI() {
        const btn = document.getElementById('ai-toggle-btn');
        if (!btn) return;
        btn.innerHTML = `<i class="fas fa-robot"></i> AI: ${this.aiEnabled ? 'ON' : 'OFF'}`;
        btn.classList.toggle('primary', this.aiEnabled);
    }
    
    resetLLMState(preserveMessage = false) {
        this.llmActionQueue = [];
        this.llmProcessing = false;
        if (!preserveMessage) {
            this.updateLLMThought('Toggle AI to let the LLM reason about moves.');
            this.updateLLMStatus('idle');
        }
    }
    
    stopLLMControl(message = 'LLM paused.', status = 'idle') {
        this.llmActionQueue = [];
        this.llmProcessing = false;
        this.updateLLMStatus(status);
        if (message) {
            this.updateLLMThought(message);
        }
    }
    
    async processLLMLoop() {
        if (!this.aiEnabled || !this.player || this.currentGoal?.completed) {
            return;
        }
        
        if (this.llmActionQueue.length > 0) {
            const nextAction = this.llmActionQueue.shift();
            await this.executeLLMTool(nextAction);
            if (this.aiEnabled) {
                setTimeout(() => this.processLLMLoop(), 450);
            }
            return;
        }
        
        if (this.llmProcessing) return;
        await this.requestLLMActions();
    }
    
    async requestLLMActions() {
        if (!this.currentGoal || !this.player) return;
        this.llmProcessing = true;
        this.updateLLMStatus('thinking');
        
        try {
            const payload = {
                goal: this.currentGoal.description,
                state: this.buildLLMStateSnapshot()
            };
            
            const response = await fetch(`${this.llmApiBaseUrl}/llm/step`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            if (!response.ok) {
                throw new Error(`Server responded with ${response.status}`);
            }
            
            const data = await response.json();
            if (data.thought) {
                this.updateLLMThought(data.thought);
            }
            if (Array.isArray(data.actions) && data.actions.length) {
                this.llmActionQueue.push(...data.actions);
            }
        } catch (error) {
            console.error('LLM request failed:', error);
            this.aiEnabled = false;
            this.updateAIToggleUI();
            this.stopLLMControl(`LLM error: ${error.message}`, 'error');
            return;
        } finally {
            this.llmProcessing = false;
        }
        
        if (this.aiEnabled) {
            setTimeout(() => this.processLLMLoop(), 350);
        }
    }
    
    buildLLMStateSnapshot() {
        return {
            gridSize: this.gridSize,
            grid: this.grid.map(row => [...row]),
            player: this.player ? {
                id: this.player.id,
                x: this.player.x,
                y: this.player.y,
                inventory: this.player.inventory.map(obj => ({ ...obj }))
            } : null,
            players: this.players.map(p => ({
                id: p.id,
                x: p.x,
                y: p.y,
                inventory: p.inventory.map(obj => ({ ...obj }))
            })),
            objects: this.objects.map(obj => ({ ...obj })),
            goalZones: this.goalZones.map(zone => ({ ...zone })),
            inventory: this.inventory.map(obj => ({ ...obj })),
            goal: this.currentGoal,
            moves: this.moves,
            score: this.score,
            roundNumber: this.roundNumber,
            timeElapsed: this.gameTime
        };
    }
    
    async executeLLMTool(action) {
        if (!action || !action.tool) return;
        const args = action.arguments || {};
        
        switch(action.tool) {
            case 'move':
                if (args.direction) {
                    this.movePlayer(args.direction);
                }
                break;
            case 'pickup':
                this.pickupObject();
                break;
            case 'drop':
                this.dropObject();
                break;
            case 'interact':
                this.interact();
                break;
            default:
                console.warn('Unknown LLM tool:', action.tool);
        }
    }
    
    updateLLMThought(text) {
        this.llmThought = text || '';
        const el = document.getElementById('llm-thought');
        if (el) {
            el.textContent = this.llmThought || 'LLM is idle.';
        }
    }
    
    updateLLMStatus(state) {
        const pill = document.getElementById('llm-status-pill');
        if (!pill) return;
        pill.classList.remove('idle', 'thinking', 'error');
        switch(state) {
            case 'thinking':
                pill.classList.add('thinking');
                pill.textContent = 'Thinking';
                break;
            case 'error':
                pill.classList.add('error');
                pill.textContent = 'Error';
                break;
            default:
                pill.classList.add('idle');
                pill.textContent = 'Idle';
        }
    }
    
    render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw grid
        for (let y = 0; y < this.gridSize; y++) {
            for (let x = 0; x < this.gridSize; x++) {
                const cellX = x * this.cellSize;
                const cellY = y * this.cellSize;
                
                // Draw cell background
                this.ctx.fillStyle = this.colors[this.grid[y][x]];
                this.ctx.fillRect(cellX, cellY, this.cellSize, this.cellSize);
                
                // Draw grid lines
                this.ctx.strokeStyle = '#333';
                this.ctx.strokeRect(cellX, cellY, this.cellSize, this.cellSize);
            }
        }
        
        // Draw goal zones
        this.goalZones.forEach(zone => {
            const x = zone.x * this.cellSize + this.cellSize / 2;
            const y = zone.y * this.cellSize + this.cellSize / 2;
            
            this.ctx.fillStyle = this.colors.goalZone[zone.color] + '40';
            this.ctx.fillRect(zone.x * this.cellSize, zone.y * this.cellSize, this.cellSize, this.cellSize);
            
            this.ctx.strokeStyle = this.colors.goalZone[zone.color];
            this.ctx.lineWidth = 3;
            this.ctx.strokeRect(zone.x * this.cellSize + 5, zone.y * this.cellSize + 5, this.cellSize - 10, this.cellSize - 10);
            this.ctx.lineWidth = 1;
        });
        
        // Draw objects
        this.objects.forEach(object => {
            const x = object.x * this.cellSize + this.cellSize / 2;
            const y = object.y * this.cellSize + this.cellSize / 2;
            
            this.ctx.fillStyle = this.colors[object.shape][object.color];
            
            if (object.shape === 'cube') {
                this.ctx.fillRect(x - 15, y - 15, 30, 30);
                this.ctx.strokeStyle = '#000';
                this.ctx.strokeRect(x - 15, y - 15, 30, 30);
            } else {
                this.ctx.beginPath();
                this.ctx.arc(x, y, 15, 0, Math.PI * 2);
                this.ctx.fill();
                this.ctx.strokeStyle = '#000';
                this.ctx.stroke();
            }
        });
        
        // Draw players
        this.players.forEach((player, index) => {
            const x = player.x * this.cellSize + this.cellSize / 2;
            const y = player.y * this.cellSize + this.cellSize / 2;
            
            this.ctx.fillStyle = player.color;
            this.ctx.beginPath();
            this.ctx.arc(x, y, 18, 0, Math.PI * 2);
            this.ctx.fill();
            
            this.ctx.strokeStyle = '#fff';
            this.ctx.lineWidth = 2;
            this.ctx.stroke();
            this.ctx.lineWidth = 1;
            
            // Draw player number
            this.ctx.fillStyle = '#fff';
            this.ctx.font = 'bold 12px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.textBaseline = 'middle';
            this.ctx.fillText(index + 1, x, y);
        });
    }
    
    updateUI() {
        if (this.gameMode === 'single') {
            document.getElementById('score').textContent = this.score;
            document.getElementById('moves').textContent = this.moves;
        } else {
            // Show both players' scores for multiplayer
            const totalScore = Object.values(this.scores).reduce((sum, score) => sum + score, 0);
            const totalMoves = Object.values(this.playerMoves).reduce((sum, moves) => sum + moves, 0);
            document.getElementById('score').textContent = `${this.scores.player1 || 0} / ${this.scores.player2 || 0}`;
            document.getElementById('moves').textContent = `${this.playerMoves.player1 || 0} / ${this.playerMoves.player2 || 0}`;
        }
    }
    
    updateTimerUI() {
        const minutes = Math.floor(this.gameTime / 60);
        const seconds = this.gameTime % 60;
        document.getElementById('timer').textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
    }
    
    updateInventoryUI() {
        const slots = document.querySelectorAll('.inventory-slot');
        slots.forEach((slot, index) => {
            if (index < this.inventory.length) {
                const object = this.inventory[index];
                slot.className = 'inventory-slot';
                slot.innerHTML = object.shape === 'cube' ? '◼' : '●';
                slot.style.color = this.colors[object.shape][object.color];
            } else {
                slot.className = 'inventory-slot empty';
                slot.innerHTML = '';
            }
        });
    }
    
    showMessage(title, message, type = 'info') {
        const messageBox = document.getElementById('message-box');
        const messageContent = messageBox.querySelector('.message-content');
        
        messageContent.innerHTML = `<h3>${title}</h3><p>${message}</p>`;
        messageBox.style.display = 'block';
        
        if (type === 'success') {
            messageBox.style.borderColor = this.colors.success;
        } else if (type === 'danger') {
            messageBox.style.borderColor = this.colors.danger;
        }
    }
    
    hideMessage() {
        document.getElementById('message-box').style.display = 'none';
    }
    
    showGameOverModal() {
        clearInterval(this.timerInterval);
        
        const modal = document.getElementById('game-over-modal');
        document.getElementById('final-score').textContent = this.score;
        document.getElementById('final-time').textContent = document.getElementById('timer').textContent;
        document.getElementById('final-moves').textContent = this.moves;
        document.getElementById('efficiency-bonus').textContent = Math.max(0, 50 - this.moves);
        
        modal.style.display = 'flex';
    }
    
    hideGameOverModal() {
        document.getElementById('game-over-modal').style.display = 'none';
    }
    
    resetGame() {
        clearInterval(this.timerInterval);
        this.startNewGame();
    }
}

// Initialize game when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.game = new MiniQuestArena();
});
