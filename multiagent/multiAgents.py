'''
Partner 1: Gurkirat Singh (A11593827)
Partner 2: Jessica Ng (A10683076)
'''

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
import random, util, sys

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

        '''We don't care how close we are to the ghost as long as we don't touch the 
        ghost. If food is closer than ghost, head towards food.'''

        finalVal = 0
        min_val = -sys.maxint - 1 

        '''We never want to stop or bump into the ghost because that will make us
        lose a lot of points. Stopping is not helpful for the pacman since it is
        better to move foward in some direction in the hope that a better
        scenario presents itself'''


        if action == "Stop" or len(filter(lambda x: (x.getPosition() == newPos and
            x.scaredTimer == 0), newGhostStates)) != 0:
            '''return the lowest int value'''
            return min_val

        '''Get the distance to all the possible foods in the grid'''
        foodDistList = map(lambda food: util.manhattanDistance(food, newPos),
            currentGameState.getFood().asList())

        foodDistList += map(lambda capsule: util.manhattanDistance(capsule, newPos),
            currentGameState.getCapsules()) 

        '''We want the closest food to the pacman'''
        if len(foodDistList) > 0:
            finalVal += min(foodDistList)

        
        ghostList = currentGameState.getGhostPositions()
        ghostDistList = map(lambda ghost: util.manhattanDistance(ghost, newPos), ghostList)
        minGhostDist = min(ghostDistList)
        

        '''Only if the ghost is closer to food, we eat the scared ghost'''
        for idx, timer in enumerate(newScaredTimes):
            if len(ghostDistList) > 0:
                val = timer - ghostDistList[idx]
                if val > 0 and finalVal > minGhostDist:
                    finalVal -= minGhostDist
        return -(finalVal)

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

        
        newDepth = self.depth 
        solutionList = []
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            solutionVal = self.getValue(successor, newDepth, 1)
            solutionList.append((solutionVal, action))
        
        return max(solutionList)[1]
        
    def getValue(self, gameState, depth, playerType):   
        if depth == 0 or len(gameState.getLegalActions(playerType)) == 0:
            return (self.evaluationFunction(gameState))
        
        legalMoves = gameState.getLegalActions(playerType)
        numAgents = gameState.getNumAgents()

        valueOfMoves = []
        if playerType == 0:
            
            for action in legalMoves:
                successor = gameState.generateSuccessor(0, action)
                val = self.getValue(successor, depth, 1)
                valueOfMoves.append(val)
            return max(valueOfMoves)

        else:
            newPlayerType = (playerType + 1) % numAgents
            depth -= (playerType + 1)/numAgents

            for action in legalMoves:
                successor = gameState.generateSuccessor(playerType, action)
                val = self.getValue(successor, depth, newPlayerType)
                valueOfMoves.append(val)

            return min(valueOfMoves)
        
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
            return (self.evaluationFunction(gameState),)
        
        legalMoves = gameState.getLegalActions(playerType)
        newPlayerType = (playerType + 1) % numOfAgents
        
        if playerType == 0:
            maxMove = (MIN, Directions.STOP)
            for action in legalMoves:
                successor = gameState.generateSuccessor(playerType, action)
                val = self.getAlphaBeta(successor, depth-1, alpha, beta, newPlayerType, numOfAgents)
                maxMove = max(maxMove, (val[0], action))
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

    DESCRIPTION: In the case that the move is STOP or the ghostPos = pacmanPos,
    we want to return the minimum value possible. This is because these are the moves
    we consider least optimal. We want to keep Pacman moving and if ghostPos = pacmanPos,
    then the game has been lost.

    In the case that there are no nearby scared ghosts, we want the action that is closest
    to the nearest food. In order to achieve this we use 1/minFoodDist. Large distances will
    yield smaller results as we use the distance as a fraction, and the smallest food distance
    will yield the highest score in this case. In this case, we subtract the 1/minGhostDist
    because we wish to maximize the distance between Pacman and the ghost when the ghost is not 
    edible.

    In the case that there are nearby scared ghosts, we prioritize this state over other states.
    We add the distance of the first scared ghost that is closer than the minimum food distance.
    This places its priority over the actions that move toward the smallest food distance as 
    there is an additional factor being added in.
    """

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    '''We don't care how close we are to the ghost as long as we don't touch the 
    ghost. If food is closer than ghost, head towards food.'''

    finalVal = 0
    min_val = -sys.maxint - 1 
    #ghost_pos = successorGameState.getGhostPositions()

    '''We never want to stop or bump into the ghost because that will make us
    lose a lot of points. Stopping is not helpful for the pacman since it is
    better to move foward in some direction in the hope that a better
    scenario presents itself'''


    if currentGameState.getPacmanState == 'STOP' or len(filter(lambda x: (x.getPosition() == newPos and
        x.scaredTimer == 0), newGhostStates)) != 0:
        '''return the lowest int value'''
        return min_val

    '''Get the distance to all the possible foods in the grid'''
    foodDistList = map(lambda food: util.manhattanDistance(food, newPos),
        currentGameState.getFood().asList())

    foodDistList += map(lambda capsule: util.manhattanDistance(capsule, newPos),
        currentGameState.getCapsules()) 

    '''We want the closest food/capsule to the pacman'''
    if len(foodDistList) > 0:
        minFoodDist = float(min(foodDistList))
        finalVal += 1.0/float(min(foodDistList))
    else:
        minFoodDist = 0.0

    '''Subtract the minimum ghost distance for now. We want to maximize distance to ghosts
    if possible'''
    ghostList = currentGameState.getGhostPositions()
    ghostDistList = map(lambda ghost: util.manhattanDistance(ghost, newPos), ghostList)
    minGhostDist =  1.0/float(min(ghostDistList))
    finalVal -= minGhostDist

    '''But if the ghost is scared and it's closer than the nearest food, undo what we just did
    and add the distance to the scared ghost to our final score'''
    for idx, timer in enumerate(newScaredTimes):
        if len(ghostDistList) > 0:
            val = timer - ghostDistList[idx]
            if val > 0 and minFoodDist > ghostDistList[idx]:
                finalVal += 1.0/float(ghostDistList[idx]) + minGhostDist
                break

    return finalVal + float(currentGameState.getScore())
       

# Abbreviation
better = betterEvaluationFunction

