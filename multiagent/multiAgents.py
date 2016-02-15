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

#using 32-bit max and min ints
MAX = 999999
MIN = -999999

MAXFLOAT = 999999.0
MINFLOAT = -999999.0

INF = 2147483647

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        print "Legal Moves: ", legalMoves

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        print "Scores: ", scores
        bestScore = max(scores)
        print "Best score: ", bestScore
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        print "Best indicies: ", bestIndices
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        print "chosenIndex: ", chosenIndex
        print "legal moves index: ", legalMoves[chosenIndex]
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

        "*** YOUR CODE HERE ***"
        print "successorGameState: ", successorGameState
        print "Ghost states: ", successorGameState.getGhostPositions()
        print "successorGameState Scores", successorGameState.getScore()
        print "newPos: ", newPos
        print "newFood: ", newFood
        print "newScaredTimes: ", newScaredTimes
        return successorGameState.getScore()

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
        """
        "*** YOUR CODE HERE ***"
        
        numOfAgents = gameState.getNumAgents()
        newDepth = self.depth * numOfAgents
        
        solution = self.getValue(gameState, newDepth, 0, numOfAgents)
        return solution[1]
        
    def getValue(self, gameState, depth, playerType, numOfAgents):   
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return (self.evaluationFunction(gameState), Directions.STOP)
        
        legalMoves = gameState.getLegalActions(playerType)
        newPlayerType = (playerType + 1) % numOfAgents
        
        valueOfMoves = []
        for action in legalMoves:
            successor = gameState.generateSuccessor(playerType, action)
            val = self.getValue(successor, depth-1, newPlayerType, numOfAgents)
            valueOfMoves.append((val, action))
        
        return min(valueOfMoves) if playerType >= 1 else max(valueOfMoves)
        
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        #Returns the minimax action using self.depth and self.evaluationFunction
        "*** YOUR CODE HERE ***"
        numOfAgents = gameState.getNumAgents()
        newDepth = self.depth * numOfAgents
        
        alpha = (MIN, Directions.STOP)
        beta = (MAX, Directions.STOP)
        solution = self.getAlphaBeta(gameState, newDepth, alpha, beta, 0, numOfAgents)
        return solution[1]
        
    def getAlphaBeta(self, gameState, depth, alpha, beta, playerType, numOfAgents):    
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return (self.evaluationFunction(gameState), Directions.STOP)
        
        legalMoves = gameState.getLegalActions(playerType)
        newPlayerType = (playerType + 1) % numOfAgents
        
        if playerType == 0:
            maxMove = (MIN, Directions.STOP)
            for action in legalMoves:
                successor = gameState.generateSuccessor(playerType, action)
                val = self.getAlphaBeta(successor, depth-1, alpha, beta, newPlayerType, numOfAgents)
                maxMove = max(maxMove, (val[0], action))
                """algo says that this comparison should be >= but the tests use > & <.
                I AM CONFUSED. THE TEST CASE IT FAILS LOOKS WRONG."""
                if maxMove[0] > beta[0]:
                    break
                alpha = max(alpha, maxMove)
            return maxMove
            
        else:
            minMove = (MAX, None)
            for action in legalMoves:
                successor = gameState.generateSuccessor(playerType, action)
                val = self.getAlphaBeta(successor, depth-1, alpha, beta, newPlayerType, numOfAgents)
                minMove = min(minMove, (val[0], action))
                """using < instead of <=???"""
                if minMove[0] < alpha[0]:
                    break
                beta = min(beta, minMove)
            return minMove        
        
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
        "*** YOUR CODE HERE ***"
        numOfAgents = gameState.getNumAgents()
        newDepth = self.depth * numOfAgents
        legalMoves = gameState.getLegalActions(0)

        expectedValue = (MINFLOAT, Directions.STOP)
        for action in legalMoves:
            successor = gameState.generateSuccessor(0, action)
            val = self.getExpectimax(successor, newDepth-1, 1, numOfAgents)
            expectedValue = max(expectedValue, (val, action))
            
        return expectedValue[1]
            
    def getExpectimax(self, gameState, depth, playerType, numOfAgents):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return float(self.evaluationFunction(gameState))
        
        legalMoves = gameState.getLegalActions(playerType)
        newPlayerType = (playerType + 1) % numOfAgents
        expectedValue = 0.0
        
        if playerType == 0:
            expectedValue = MINFLOAT
            for action in legalMoves:
                successor = gameState.generateSuccessor(playerType, action)
                val = self.getExpectimax(successor, depth-1, newPlayerType, numOfAgents)
                expectedValue = max(expectedValue, val)
        else:
            for action in legalMoves:
                successor = gameState.generateSuccessor(playerType, action)
                val = self.getExpectimax(successor, depth-1, newPlayerType, numOfAgents)
                expectedValue += 1.0/float(len(legalMoves)) * val
        return expectedValue
                
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

