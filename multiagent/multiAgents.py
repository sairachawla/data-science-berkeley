# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        """goal: eat food pellets, avoid ghosts

        take reciprocal since higher scores = better evaluation"""
        foodList = newFood.asList()
        score = 0

        for f in foodList:
            d = abs(util.manhattanDistance(f, newPos))
            if d > 0:
                score = score + (1.0/d)

        for g in newGhostStates:
            gPos = g.getPosition()
            d = abs(util.manhattanDistance(gPos, newPos))
            if d > 1:
                score = score + (1.0/d)

        return successorGameState.getScore() + score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        def norm(gameState, agentInd, depth):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            if (agentInd - gameState.getNumAgents()) >= 0:
                return maximize(gameState, 0, depth)
            else:
                return minimize(gameState, agentInd, depth)


        def minimize(gameState, numGhosts, depth):
            actions = gameState.getLegalActions(numGhosts)
            score = float('inf')
            moves = depth

            #checks if we are on the last ghost
            if numGhosts == gameState.getNumAgents() -1:
                moves -= 1

            for a in actions:
                newstate = gameState.generateSuccessor(numGhosts, a)
                value = norm(newstate, numGhosts + 1, moves)
                if value < score:
                    score = value
            return score

        def maximize(gameState, agentInd, depth):
            score = float('-inf')
            actions = gameState.getLegalActions(agentInd)
            moves = depth
            
            for a in actions:
                s = gameState.generateSuccessor(agentInd, a)
                value = norm(s, agentInd + 1, moves)
                if value > score:
                    score = value
            return score



        acs = gameState.getLegalActions(self.index)
        final_Score = float('-inf')
        final_Action = None

        for a in acs:
            state = gameState.generateSuccessor(self.index, a)
            value = norm(state, self.index + 1, self.depth)
            if value > final_Score:
                final_Score = value
                final_Action = a
        return final_Action

        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        alpha = float('-inf') #lower bound
        beta = float('inf') #upper bound

        def norm(gameState, agentInd, depth, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            if (agentInd - gameState.getNumAgents()) >= 0:
                return maximize(gameState, 0, depth, alpha, beta)
            else:
                return minimize(gameState, agentInd, depth, alpha, beta)


        def minimize(gameState, numGhosts, depth, alpha, beta):
            actions = gameState.getLegalActions(numGhosts)
            score = float('inf')
            moves = depth

            #checks if we are on the last ghost
            if numGhosts == gameState.getNumAgents() -1:
                moves -= 1

            for a in actions:
                newstate = gameState.generateSuccessor(numGhosts, a)
                value = norm(newstate, numGhosts + 1, moves, alpha, beta)
                if value < score:
                    score = value
                if value < alpha:
                    return value
                if value < beta:
                    beta = value
            return score

        def maximize(gameState, agentInd, depth, alpha, beta):
            score = float('-inf')
            actions = gameState.getLegalActions(agentInd)
            moves = depth
            
            for a in actions:
                s = gameState.generateSuccessor(agentInd, a)
                value = norm(s, agentInd + 1, moves, alpha, beta)
                if value > score:
                    score = value
                if value > beta:
                    return value
                if value > alpha:
                    alpha = value
            return score



        acs = gameState.getLegalActions(self.index)
        final_Score = float('-inf')
        final_Action = None

        for a in acs:
            state = gameState.generateSuccessor(self.index, a)
            value = norm(state, self.index+1, self.depth, alpha, beta)
            if value > final_Score:
                final_Score = value
                final_Action = a
            if value > beta:
                return action
            if value > alpha:
                alpha = value
        return final_Action
        
    

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def norm(gameState, agentInd, depth):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            if (agentInd - gameState.getNumAgents()) >= 0:
                return maximize(gameState, 0, depth)
            else:
                return minimize(gameState, agentInd, depth)


        def minimize(gameState, numGhosts, depth):
            actions = gameState.getLegalActions(numGhosts)
            values = 0
            score = float('inf')
            moves = depth

            #checks if we are on the last ghost
            if numGhosts == gameState.getNumAgents() -1:
                moves -= 1

            for a in actions:
                newstate = gameState.generateSuccessor(numGhosts, a)
                values += norm(newstate, numGhosts + 1, moves)
                score = values/len(actions)
            return score

        def maximize(gameState, agentInd, depth):
            score = float('-inf')
            actions = gameState.getLegalActions(agentInd)
            moves = depth
            
            for a in actions:
                s = gameState.generateSuccessor(agentInd, a)
                value = norm(s, agentInd + 1, moves)
                if value > score:
                    score = value
            return score



        acs = gameState.getLegalActions(self.index)
        final_Score = float('-inf')
        final_Action = None

        for a in acs:
            state = gameState.generateSuccessor(self.index, a)
            value = norm(state, self.index+1, self.depth)
            if value > final_Score:
                final_Score = value
                final_Action = a
        return final_Action


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    Raises Score: 
    - distance to closest food pellet
    - distance to nearest scared ghost 

    Reduces Score:
    - distance to nearest ghost, BUT LIKE #1, we want to stay one step ahead of the 
        ghost. 

    Weigh each of these factors?

    """
    pacPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    foods = food.asList()

    if currentGameState.isWin():
        return float('inf') #high score is better
    elif currentGameState.isLose():
        return float('-inf') #lower score bc we lost :(
    else:    #neither losing or winning yet

        
        #calculate distance to the nearest ghost
        closest_g = 1
        ghostsDist =  [manhattanDistance(pacPos, g.getPosition()) for g in ghostStates if not g.scaredTimer]
        if len(ghostsDist) > 0:
            closest_g = min(ghostsDist)

        #calculate distance to nearest SCARED ghost
        scaredGhostsDist = []
        closest_scared_g = 1
        for scared_g in ghostStates:
            if scared_g.scaredTimer > 0:
                scaredGhostsDist.append(manhattanDistance(pacPos, scared_g.getPosition()))
        if len(scaredGhostsDist) > 0:
            closest_scared_g = min(scaredGhostsDist)

        #calculate distance to the nearest food pellet
        foodsDist = []
        for f in foods:
            foodsDist.append(manhattanDistance(pacPos,f))
        closest_f = min(foodsDist)

        return currentGameState.getScore() - 2 * len(foods) + 4 * (1/closest_f) - (1/closest_g) - 2 * (1/closest_scared_g)





    
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
