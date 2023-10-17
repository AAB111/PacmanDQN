import pygame
import math
from random import randrange
import random
import copy,threading
import numpy as np
board_path = "Assets/BoardImages/"
element_path = "Assets/ElementImages/"
text_path = "Assets/TextImages/"
data_path = "Assets/Data/"
music_path = "Assets/Music/"

pygame.mixer.init()
pygame.init()
print(pygame.mixer.music.get_busy())

# 28 Across 31 Tall 1: Empty Space 2: Tic-Tac 3: Wall 4: Ghost safe-space 5: Special Tic-Tac 7: Pacman
original_game_board = [
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
    [3,2,2,2,2,2,2,2,2,2,2,2,2,3,3,2,2,2,2,2,2,2,2,2,2,2,2,3],
    [3,2,3,3,3,3,2,3,3,3,3,3,2,3,3,2,3,3,3,3,3,2,3,3,3,3,2,3],
    [3,6,3,3,3,3,2,3,3,3,3,3,2,3,3,2,3,3,3,3,3,2,3,3,3,3,6,3],
    [3,2,3,3,3,3,2,3,3,3,3,3,2,3,3,2,3,3,3,3,3,2,3,3,3,3,2,3],
    [3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3],
    [3,2,3,3,3,3,2,3,3,2,3,3,3,3,3,3,3,3,2,3,3,2,3,3,3,3,2,3],
    [3,2,3,3,3,3,2,3,3,2,3,3,3,3,3,3,3,3,2,3,3,2,3,3,3,3,2,3],
    [3,2,2,2,2,2,2,3,3,2,2,2,2,3,3,2,2,2,2,3,3,2,2,2,2,2,2,3],
    [3,3,3,3,3,3,2,3,3,3,3,3,1,3,3,1,3,3,3,3,3,2,3,3,3,3,3,3],
    [3,3,3,3,3,3,2,3,3,3,3,3,1,3,3,1,3,3,3,3,3,2,3,3,3,3,3,3],
    [3,3,3,3,3,3,2,3,3,1,1,1,1,1,1,1,1,1,1,3,3,2,3,3,3,3,3,3],
    [3,3,3,3,3,3,2,3,3,1,3,3,3,3,3,3,3,3,1,3,3,2,3,3,3,3,3,3],
    [3,3,3,3,3,3,2,3,3,1,3,4,4,4,4,4,4,3,1,3,3,2,3,3,3,3,3,3],
    [1,1,1,1,1,1,2,1,1,1,3,4,4,4,4,4,4,3,1,1,1,2,1,1,1,1,1,1], # Middle Lane Row: 14
    [3,3,3,3,3,3,2,3,3,1,3,4,4,4,4,4,4,3,1,3,3,2,3,3,3,3,3,3],
    [3,3,3,3,3,3,2,3,3,1,3,3,3,3,3,3,3,3,1,3,3,2,3,3,3,3,3,3],
    [3,3,3,3,3,3,2,3,3,1,1,1,1,1,1,1,1,1,1,3,3,2,3,3,3,3,3,3],
    [3,3,3,3,3,3,2,3,3,1,3,3,3,3,3,3,3,3,1,3,3,2,3,3,3,3,3,3],
    [3,3,3,3,3,3,2,3,3,1,3,3,3,3,3,3,3,3,1,3,3,2,3,3,3,3,3,3],
    [3,2,2,2,2,2,2,2,2,2,2,2,2,3,3,2,2,2,2,2,2,2,2,2,2,2,2,3],
    [3,2,3,3,3,3,2,3,3,3,3,3,2,3,3,2,3,3,3,3,3,2,3,3,3,3,2,3],
    [3,2,3,3,3,3,2,3,3,3,3,3,2,3,3,2,3,3,3,3,3,2,3,3,3,3,2,3],
    [3,6,2,2,3,3,2,2,2,2,2,2,2,7,2,2,2,2,2,2,2,2,3,3,2,2,5,3],
    [3,3,3,2,3,3,2,3,3,2,3,3,3,3,3,3,3,3,2,3,3,2,3,3,2,3,3,3],
    [3,3,3,2,3,3,2,3,3,2,3,3,3,3,3,3,3,3,2,3,3,2,3,3,2,3,3,3],
    [3,2,2,2,2,2,2,3,3,2,2,2,2,3,3,2,2,2,2,3,3,2,2,2,2,2,2,3],
    [3,2,3,3,3,3,3,3,3,3,3,3,2,3,3,2,3,3,3,3,3,3,3,3,3,3,2,3],
    [3,2,3,3,3,3,3,3,3,3,3,3,2,3,3,2,3,3,3,3,3,3,3,3,3,3,2,3],
    [3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3],
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
]

game_board = copy.deepcopy(original_game_board)
sprite_ratio = 3/2
square = 20 # Size of each unit square
sprite_offset = square * (1 - sprite_ratio) * (1/2)
(width, height) = (len(game_board[0]) * square, len(game_board) * square) # Game screen
screen = pygame.display.set_mode((width, height))
pygame.display.update()
# pelletColor = (165, 93, 53)
pellet_color = (222, 161, 133)

PLAYING_KEYS = {
    "up":[pygame.K_w, pygame.K_UP],
    "down":[pygame.K_s, pygame.K_DOWN],
    "right":[pygame.K_d, pygame.K_RIGHT],
    "left":[pygame.K_a, pygame.K_LEFT]
}

class Game:
    def __init__(self, level, score):
        self.paused = False
        self.number_states = {'row':36,'col':28}
        self.ghost_update_delay = 1
        self.ghost_update_count = 0
        self.pacman_update_delay = 1
        self.pacman_update_count = 0
        self.tictak_change_delay = 10
        self.tictak_change_count = 0
        self.ghosts_attacked = False
        self.high_score = self.getHighScore()
        self.score = score
        self.level = level
        self.lives = 3
        # self.ghosts = [Ghost(14, 13, "red", 0), Ghost(17, 11, "blue", 1), Ghost(17, 13, "pink", 2), Ghost(17, 15, "orange", 3)]
        self.ghosts = []
        self.pacman = Pacman(26,13) # Center of Second Last Row
        self.total = self.getCountPoints()
        self.ghost_score = 200
        self.levels = [[350, 250], [150, 450], [150, 450], [0, 600]]
        random.shuffle(self.levels)
        # Level index and Level Progress
        self.ghost_states = [[1, 0], [0, 0], [1, 0], [0, 0]]
        index = 0
        for state in self.ghost_states:
            state[0] = randrange(2)
            state[1] = randrange(self.levels[index][state[0]] + 1)
            index += 1
        self.collected = 0
        self.started = True
        self.game_over = False
        self.game_over_counter = 0
        self.points = []
        self.points_timer = 10
        # Berry Spawn Time, Berry Death Time, Berry Eaten
        self.berry_state = [200, 400, False]
        self.berry_location = [20.0, 13]
        self.berries = ["tile080.png", "tile081.png", "tile082.png", "tile083.png", "tile084.png", "tile085.png", "tile086.png", "tile087.png"]
        self.berries_collected = []
        self.level_timer = 0
        self.berry_score = 100
        self.locked_in_timer = 100
        self.locked_in = True
        self.extra_life_given = False
        self.FPS = 15
        self.playtime = self.FPS * 120
        self.timer_game = 0
        self.time_end_level = self.timer_game
        self.reward = 'Empty'
        self.rewards = {'Empty' : -1, #Empty cell
                        'TicTac' : 3,# Tic-tac
                        'Wall' : -50,#Wall
                        'Wall' : -50,#Ghost safe-zone
                        'BigTicTac' : 50,#Big tic-tac
                        'BigTicTac' : 50,#Big tic-tac
                        'Win' : 1000,#Win Collected all tic-tac
                        'LoseRound' : -30,#Lose Round
                        'GameOver' : -1000} # GameOver

    def newGame(self):
        global game_board 
        game_board = copy.deepcopy(original_game_board)

    def exponenta(self,x):
        return math.log10(math.exp(x))

    def rewardTicTac(self):
        return self.collected * 5

    def getReward(self):
        if self.reward == 'Win':
            return self.getRewardWin()
        if self.checkPlaytimeOver():
            return self.getRewardGameOver()
        if self.pacman.reward == 'Wall':
            return self.rewards[self.pacman.reward]
        if self.reward == 'TicTac' or self.reward == 'BigTicTac':
            return self.rewardTicTac()
        return self.rewards[self.reward]

    def getStatePacman(self):
        row = round(self.pacman.row)
        col = round(self.pacman.col)
        return (row,col)

    def getStateEnv(self):
        return game_board

    def step(self,action):
        self.pacman.newDir = action
        clock = pygame.time.Clock() 
        clock.tick(self.FPS)
        time_game = self.timerGame()
        self.update()
        next_state = self.getStateEnv()
        if(self.reward == 'Win' or self.reward == 'GameOver'):
            done = True
        else:
            done = False
        reward = self.getReward()
        self.reward = 'Empty'
        self.pacman.reward = ''
        return (next_state,reward,done,time_game)

    def getRewardWin(self):
        return self.rewards['Win'] * (self.playtime / self.time_end_level)

    def getRewardGameOver(self):
        return  self.rewards['GameOver'] - (self.rewards['GameOver'] * (self.collected / self.total))

    def timerGame(self):
        return float(self.timer_game / self.playtime)

    def checkPlaytimeOver(self):
        return self.timerGame() >= 1.0
    # Driver method: The games primary update method
    def update(self):
        # pygame.image.unload()
        if self.game_over or self.checkPlaytimeOver():
            self.reward = 'GameOver'
            self.gameOverFunc()
            return
        self.level_timer += 1
        self.timer_game += 1
        self.ghost_update_count += 1
        self.pacman_update_count += 1
        self.tictak_change_count += 1
        self.ghosts_attacked = False
        # Draw tiles around ghosts and pacman
        self.clearBoard()
        for ghost in self.ghosts:
            if ghost.attacked:
                self.ghosts_attacked = True

        # Check if the ghost should case pacman
        index = 0
        for state in self.ghost_states:
            state[1] += 1
            if state[1] >= self.levels[index][state[0]]:
                state[1] = 0
                state[0] += 1
                state[0] %= 2
            index += 1

        index = 0
        for ghost in self.ghosts:
            if not ghost.attacked and not ghost.dead and self.ghost_states[index][0] == 0:
                ghost.target = [self.pacman.row, self.pacman.col]
            index += 1

        if self.level_timer == self.locked_in_timer:
            self.locked_in = False

        self.checkSurroundings()
        if self.ghost_update_count == self.ghost_update_delay:
            for ghost in self.ghosts:
                ghost.update()
            self.ghost_update_count = 0

        if self.tictak_change_count == self.tictak_change_delay:
            #Changes the color of special Tic-Taks
            self.flipColor()
            self.tictak_change_count = 0

        if self.pacman_update_count == self.pacman_update_delay:
            self.pacman_update_count = 0
            oldPosition = (self.pacman.row, self.pacman.col)
            self.pacman.update()
            newPosition = (self.pacman.row, self.pacman.col)
            self.pacman.col %= len(game_board[0])
            if self.pacman.row % 1.0 == 0 and self.pacman.col % 1.0 == 0:
                if game_board[int(self.pacman.row)][int(self.pacman.col)] == 2:
                    game_board[int(self.pacman.row)][int(self.pacman.col)] = 7
                    game_board[oldPosition[0]][oldPosition[1]] = 1
                    self.score += 20
                    self.collected += 1
                    self.reward = 'TicTac'
                    # Fill tile with black
                    pygame.draw.rect(screen, (0, 0, 0), (self.pacman.col * square, self.pacman.row * square, square, square))
                elif game_board[int(self.pacman.row)][int(self.pacman.col)] == 5 or game_board[int(self.pacman.row)][int(self.pacman.col)] == 6:
                    game_board[int(self.pacman.row)][int(self.pacman.col)] = 7
                    game_board[oldPosition[0]][oldPosition[1]] = 1
                    self.collected += 1
                    # Fill tile with black
                    self.reward = 'BigTicTac'
                    pygame.draw.rect(screen, (0, 0, 0), (self.pacman.col * square, self.pacman.row * square, square, square))
                    self.score += 50
                    self.ghost_score = 200
                    for ghost in self.ghosts:
                        ghost.attackedCount = 0
                        ghost.setAttacked(True)
                        ghost.setTarget()
                        self.ghosts_attacked = True
                elif game_board[int(self.pacman.row)][int(self.pacman.col)] == 1:
                    game_board[int(self.pacman.row)][int(self.pacman.col)] = 7
                    game_board[oldPosition[0]][oldPosition[1]] = 1
        self.checkSurroundings()
        self.high_score = max(self.score, self.high_score)

        global running
        if self.collected == self.total:
            print("New Level")
            self.reward = 'Win'
            self.time_end_level = self.timer_game
            self.newLevel()

        self.softRender()
    
    def pause(self,time):
        cur = 0
        while not cur == time:
            cur += 1
        # Render method

    ## отрисовка поля точек призраков пакмана
    def render(self):
        screen.fill((0, 0, 0)) # Flushes the screen
        # Draws game elements
        currentTile = 0
        self.displayLives()
        self.displayScore()
        for i in range(3, len(game_board) - 2):
            for j in range(len(game_board[0])):
                if game_board[i][j] == 3: # Draw wall
                    imageName = str(currentTile)
                    if len(imageName) == 1:
                        imageName = "00" + imageName
                    elif len(imageName) == 2:
                         imageName = "0" + imageName
                    # Get image of desired tile
                    imageName = "tile" + imageName + ".png"
                    tileImage = pygame.image.load(board_path + imageName)
                    tileImage = pygame.transform.scale(tileImage, (square, square))
                    #Display image of tile
                    screen.blit(tileImage, (j * square, i * square, square, square))

                    # pygame.draw.rect(screen, (0, 0, 255),(j * square, i * square, square, square)) # (x, y, width, height)
                elif game_board[i][j] == 2: # Draw Tic-Tak
                    pygame.draw.circle(screen, pellet_color,(j * square + square//2, i * square + square//2), square//4)
                elif game_board[i][j] == 5: #Black Special Tic-Tak
                    pygame.draw.circle(screen, (0, 0, 0),(j * square + square//2, i * square + square//2), square//2)
                elif game_board[i][j] == 6: #White Special Tic-Tak
                    pygame.draw.circle(screen, pellet_color,(j * square + square//2, i * square + square//2), square//2)
                currentTile += 1
        # Draw Sprites
        for ghost in self.ghosts:
            ghost.draw()
        self.pacman.draw()
        # Updates the screen
        pygame.display.update()

    def softRender(self):
        pointsToDraw = []
        for point in self.points:
            if point[3] < self.points_timer:
                pointsToDraw.append([point[2], point[0], point[1]])
                point[3] += 1
            else:
                self.points.remove(point)
                self.drawTilesAround(point[0], point[1])

        for point in pointsToDraw:
            self.drawPoints(point[0], point[1], point[2])

        # Draw Sprites
        for ghost in self.ghosts:
            ghost.draw()
        self.pacman.draw()
        self.displayScore()
        self.displayBerries()
        self.displayLives()
        # for point in pointsToDraw:
        #     self.drawPoints(point[0], point[1], point[2])
        self.drawBerry()
        # Updates the screen
        pygame.display.update()

    def writeGameBoard(self):
        np.savetxt('Data/gameBoard.txt',np.asarray(game_board,dtype=int),fmt='%d',delimiter=' ')

    def clearBoard(self):
            # Draw tiles around ghosts and pacman
            for ghost in self.ghosts:
                self.drawTilesAround(ghost.row, ghost.col)
            self.drawTilesAround(self.pacman.row, self.pacman.col)
            self.drawTilesAround(self.berry_location[0], self.berry_location[1])
            # Clears Ready! Label
            self.drawTilesAround(20, 10)
            self.drawTilesAround(20, 11)
            self.drawTilesAround(20, 12)
            self.drawTilesAround(20, 13)
            self.drawTilesAround(20, 14)

    def checkSurroundings(self):
        # Check if pacman got killed
        for ghost in self.ghosts:
            if self.touchingPacman(ghost.row, ghost.col) and not ghost.isAttacked():
                if self.lives == 1:
                    print("You lose")
                    self.game_over = True
                    #Removes the ghosts from the screen
                    for ghost in self.ghosts:
                        self.drawTilesAround(ghost.row, ghost.col)
                    self.drawTilesAround(self.pacman.row, self.pacman.col)
                    self.pacman.draw()
                    pygame.display.update()
                    self.pause(10000000)
                    return
                self.started = False
                self.reward = 'LoseRound'
                self.reset()
            elif self.touchingPacman(ghost.row, ghost.col) and ghost.isAttacked() and not ghost.isDead():
                ghost.setDead(True)
                ghost.setTarget()
                ghost.ghostSpeed = 1
                ghost.row = math.floor(ghost.row)
                ghost.col = math.floor(ghost.col)
                self.score += self.ghost_score
                self.points.append([ghost.row, ghost.col, self.ghost_score, 0])
                self.ghost_score *= 2
                self.pause(10000000)
        if self.touchingPacman(self.berry_location[0], self.berry_location[1]) and not self.berry_state[2] and self.level_timer in range(self.berry_state[0], self.berry_state[1]):
            self.berry_state[2] = True
            self.score += self.berry_score
            self.points.append([self.berry_location[0], self.berry_location[1], self.berry_score, 0])
            self.berries_collected.append(self.berries[(self.level - 1) % 8])
    # Displays the current score
    def displayScore(self):
        textOneUp = ["tile033.png", "tile021.png", "tile016.png"]
        textHighScore = ["tile007.png", "tile008.png", "tile006.png", "tile007.png", "tile015.png", "tile019.png", "tile002.png", "tile014.png", "tile018.png", "tile004.png"]
        index = 0
        scoreStart = 5
        highScoreStart = 11
        for i in range(scoreStart, scoreStart+len(textOneUp)):
            tileImage = pygame.image.load(text_path + textOneUp[index])
            tileImage = pygame.transform.scale(tileImage, (square, square))
            screen.blit(tileImage, (i * square, 4, square, square))
            index += 1
        score = str(self.score)
        if score == "0":
            score = "00"
        index = 0
        for i in range(0, len(score)):
            digit = int(score[i])
            tileImage = pygame.image.load(text_path + "tile0" + str(32 + digit) + ".png")
            tileImage = pygame.transform.scale(tileImage, (square, square))
            screen.blit(tileImage, ((scoreStart + 2 + index) * square, square + 4, square, square))
            index += 1

        index = 0
        for i in range(highScoreStart, highScoreStart+len(textHighScore)):
            tileImage = pygame.image.load(text_path + textHighScore[index])
            tileImage = pygame.transform.scale(tileImage, (square, square))
            screen.blit(tileImage, (i * square, 4, square, square))
            index += 1

        highScore = str(self.high_score)
        if highScore == "0":
            highScore = "00"
        index = 0
        for i in range(0, len(highScore)):
            digit = int(highScore[i])
            tileImage = pygame.image.load(text_path + "tile0" + str(32 + digit) + ".png")
            tileImage = pygame.transform.scale(tileImage, (square, square))
            screen.blit(tileImage, ((highScoreStart + 6 + index) * square, square + 4, square, square))
            index += 1

    def drawBerry(self):
        if self.level_timer in range(self.berry_state[0], self.berry_state[1]) and not self.berry_state[2]:
            # print("here")
            berryImage = pygame.image.load(element_path + self.berries[(self.level - 1) % 8])
            berryImage = pygame.transform.scale(berryImage, (int(square * sprite_ratio), int(square * sprite_ratio)))
            screen.blit(berryImage, (self.berry_location[1] * square, self.berry_location[0] * square, square, square))
    # Reset after death
    def reset(self):
        # self.ghosts = [Ghost(14, 13, "red", 0), Ghost(17, 11, "blue", 1), Ghost(17, 13, "pink", 2), Ghost(17, 15, "orange", 3)]
        self.ghosts = []
        for ghost in self.ghosts:
            ghost.setTarget()
        self.pacman = Pacman(26, 13)
        self.lives -= 1
        self.paused = True
        self.render()

    def drawPoints(self, points, row, col):
        pointStr = str(points)
        index = 0
        for i in range(len(pointStr)):
            digit = int(pointStr[i])
            tileImage = pygame.image.load(text_path + "tile" + str(224 + digit) + ".png")
            tileImage = pygame.transform.scale(tileImage, (square//2, square//2))
            screen.blit(tileImage, ((col) * square + (square//2 * index), row * square - 20, square//2, square//2))
            index += 1

    def drawReady(self):
        ready = ["tile274.png", "tile260.png", "tile256.png", "tile259.png", "tile281.png", "tile283.png"]
        for i in range(len(ready)):
            letter = pygame.image.load(text_path + ready[i])
            letter = pygame.transform.scale(letter, (int(square), int(square)))
            screen.blit(letter, ((11 + i) * square, 20 * square, square, square))

    def gameOverFunc(self):
        global running
        if self.game_over_counter == 12:
            running = False
            self.recordHighScore()
            return

        # Resets the screen around pacman
        self.drawTilesAround(self.pacman.row, self.pacman.col)

        # Draws new image
        pacmanImage = pygame.image.load(element_path + "tile" + str(116 + self.game_over_counter) + ".png")
        pacmanImage = pygame.transform.scale(pacmanImage, (int(square * sprite_ratio), int(square * sprite_ratio)))
        screen.blit(pacmanImage, (self.pacman.col * square + sprite_offset, self.pacman.row * square + sprite_offset, square, square))
        pygame.display.update()
        self.game_over_counter += 1

    def displayLives(self):
        # 33 rows || 28 cols
        # Lives[[31, 5], [31, 3], [31, 1]]
        livesLoc = [[34, 3], [34, 1]]
        for i in range(self.lives - 1):
            lifeImage = pygame.image.load(element_path + "tile054.png")
            lifeImage = pygame.transform.scale(lifeImage, (int(square * sprite_ratio), int(square * sprite_ratio)))
            screen.blit(lifeImage, (livesLoc[i][1] * square, livesLoc[i][0] * square - sprite_offset, square, square))

    def displayBerries(self):
        firstBerrie = [34, 26]
        for i in range(len(self.berries_collected)):
            berrieImage = pygame.image.load(element_path + self.berries_collected[i])
            berrieImage = pygame.transform.scale(berrieImage, (int(square * sprite_ratio), int(square * sprite_ratio)))
            screen.blit(berrieImage, ((firstBerrie[1] - (2*i)) * square, firstBerrie[0] * square + 5, square, square))

    def touchingPacman(self, row, col):
        if row - 0.5 <= self.pacman.row and row >= self.pacman.row and col == self.pacman.col:
            return True
        elif row + 0.5 >= self.pacman.row and row <= self.pacman.row and col == self.pacman.col:
            return True
        elif row == self.pacman.row and col - 0.5 <= self.pacman.col and col >= self.pacman.col:
            return True
        elif row == self.pacman.row and col + 0.5 >= self.pacman.col and col <= self.pacman.col:
            return True
        elif row == self.pacman.row and col == self.pacman.col:
            return True
        return False

    def newLevel(self):
        self.reset()
        self.lives += 1
        self.collected = 0
        self.started = True
        self.berry_state = [200, 400, False]
        self.level_timer = 0
        self.timer_game = 0
        self.locked_in = True
        for level in self.levels:
            level[0] = min((level[0] + level[1]) - 100, level[0] + 50)
            level[1] = max(100, level[1] - 50)
        random.shuffle(self.levels)
        index = 0
        for state in self.ghost_states:
            state[0] = randrange(2)
            state[1] = randrange(self.levels[index][state[0]] + 1)
            index += 1
        global game_board
        game_board = copy.deepcopy(original_game_board)
        self.render()

    def getIndexStateEnvFromCoordinat(self,state):
        index = state[0] * self.number_states['col'] + state[1]
        return index

    def drawTilesAround(self, row, col):
        row = math.floor(row)
        col = math.floor(col)
        for i in range(row-2, row+3):
            for j in range(col-2, col+3):
                if i >= 3 and i < len(game_board) - 2 and j >= 0 and j < len(game_board[0]):
                    imageName = str(((i - 3) * len(game_board[0])) + j)
                    if len(imageName) == 1:
                        imageName = "00" + imageName
                    elif len(imageName) == 2:
                         imageName = "0" + imageName
                    # Get image of desired tile
                    imageName = "tile" + imageName + ".png"
                    tileImage = pygame.image.load(board_path + imageName)
                    tileImage = pygame.transform.scale(tileImage, (square, square))
                    #Display image of tile
                    screen.blit(tileImage, (j * square, i * square, square, square))

                    if game_board[i][j] == 2: # Draw Tic-Tak
                        pygame.draw.circle(screen, pellet_color,(j * square + square//2, i * square + square//2), square//4)
                    elif game_board[i][j] == 5: #Black Special Tic-Tak
                        pygame.draw.circle(screen, (0, 0, 0),(j * square + square//2, i * square + square//2), square//2)
                    elif game_board[i][j] == 6: #White Special Tic-Tak
                        pygame.draw.circle(screen, pellet_color,(j * square + square//2, i * square + square//2), square//2)
    # Flips Color of Special Tic-Taks
    def flipColor(self):
        global game_board
        for i in range(3, len(game_board) - 2):
            for j in range(len(game_board[0])):
                if game_board[i][j] == 5:
                    game_board[i][j] = 6
                    pygame.draw.circle(screen, pellet_color,(j * square + square//2, i * square + square//2), square//2)
                elif game_board[i][j] == 6:
                    game_board[i][j] = 5
                    pygame.draw.circle(screen, (0, 0, 0),(j * square + square//2, i * square + square//2), square//2)

    def getCountPoints(self):
        total = 0
        for i in range(3, len(original_game_board) - 2):
            for j in range(len(original_game_board[0])):
                if original_game_board[i][j] == 2 or original_game_board[i][j] == 5 or original_game_board[i][j] == 6:
                    total += 1
        return total

    def getHighScore(self):
        file = open(data_path + "HighScore.txt", "r")
        highScore = int(file.read())
        file.close()
        return highScore

    def recordHighScore(self):
        file = open(data_path + "HighScore.txt", "w").close()
        file = open(data_path + "HighScore.txt", "w+")
        file.write(str(self.high_score))
        file.close()

class Pacman:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.mouthOpen = False
        self.pacSpeed = 1
        self.mouth_change_delay = 5
        self.mouth_change_count = 0
        self.dir = 0 # 0: North, 1: East, 2: South, 3: West
        self.oldDir = self.dir
        self.newDir = 0
        self.number_actions = 4
        self.reward = ''

    def canMove(self,row, col):
        if col == -1 or col == len(game_board[0]):
            return True
        if game_board[int(row)][int(col)] != 3:
            return True
        self.reward = 'Wall'
        return False
            
    def update(self):
        if self.newDir == 0:
            if self.canMove(math.floor(self.row - self.pacSpeed), self.col) and self.col % 1.0 == 0:
                self.row -= self.pacSpeed
                self.dir = self.newDir
                return
        elif self.newDir == 1:
            if self.canMove(self.row, math.ceil(self.col + self.pacSpeed)) and self.row % 1.0 == 0:
                self.col += self.pacSpeed
                self.dir = self.newDir
                return
        elif self.newDir == 2:
            if self.canMove(math.ceil(self.row + self.pacSpeed), self.col) and self.col % 1.0 == 0:
                self.row += self.pacSpeed
                self.dir = self.newDir
                return
        elif self.newDir == 3:
            if self.canMove(self.row, math.floor(self.col - self.pacSpeed)) and self.row % 1.0 == 0:
                self.col -= self.pacSpeed
                self.dir = self.newDir
                return
        if self.dir == 0:
            if self.canMove(math.floor(self.row - self.pacSpeed), self.col) and self.col % 1.0 == 0:
                self.row -= self.pacSpeed
        elif self.dir == 1:
            if self.canMove(self.row, math.ceil(self.col + self.pacSpeed)) and self.row % 1.0 == 0:
                self.col += self.pacSpeed
        elif self.dir == 2:
            if self.canMove(math.ceil(self.row + self.pacSpeed), self.col) and self.col % 1.0 == 0:
                self.row += self.pacSpeed
        elif self.dir == 3:
            if self.canMove(self.row, math.floor(self.col - self.pacSpeed)) and self.row % 1.0 == 0:
                self.col -= self.pacSpeed
    # Draws pacman based on his current state
    def draw(self):
        if not game.started:
            pacman_image = pygame.image.load(element_path + "tile112.png")
            pacman_image = pygame.transform.scale(pacman_image, (int(square * sprite_ratio), int(square * sprite_ratio)))
            screen.blit(pacman_image, (self.col * square + sprite_offset, self.row * square + sprite_offset, square, square))
            return

        if self.mouth_change_count == self.mouth_change_delay:
            self.mouth_change_count = 0
            self.mouthOpen = not self.mouthOpen
        self.mouth_change_count += 1
        # pacmanImage = pygame.image.load("Sprites/tile049.png")
        if self.dir == 0:
            if self.mouthOpen:
                pacman_image = pygame.image.load(element_path + "tile049.png")
            else:
                pacman_image = pygame.image.load(element_path + "tile051.png")
        elif self.dir == 1:
            if self.mouthOpen:
                pacman_image = pygame.image.load(element_path + "tile052.png")
            else:
                pacman_image = pygame.image.load(element_path + "tile054.png")
        elif self.dir == 2:
            if self.mouthOpen:
                pacman_image = pygame.image.load(element_path + "tile053.png")
            else:
                pacman_image = pygame.image.load(element_path + "tile055.png")
        elif self.dir == 3:
            if self.mouthOpen:
                pacman_image = pygame.image.load(element_path + "tile048.png")
            else:
                pacman_image = pygame.image.load(element_path + "tile050.png")

        pacman_image = pygame.transform.scale(pacman_image, (int(square * sprite_ratio), int(square * sprite_ratio)))
        screen.blit(pacman_image, (self.col * square + sprite_offset, self.row * square + sprite_offset, square, square))


class Ghost:
    def __init__(self, row, col, color, changeFeetCount):
        self.row = row
        self.col = col
        self.attacked = False
        self.color = color
        self.dir = randrange(4)
        self.dead = False
        self.changeFeetCount = changeFeetCount
        self.changeFeetDelay = 5
        self.target = [-1, -1]
        self.ghostSpeed = 1/4
        self.lastLoc = [-1, -1]
        self.attackedTimer = 240
        self.attackedCount = 0
        self.deathTimer = 120
        self.deathCount = 0
        self.ghostGate = [[15, 13], [15, 14]]

    def update(self):
        # print(self.row, self.col)
        if self.target == [-1, -1] or (self.row == self.target[0] and self.col == self.target[1]) or game_board[int(self.row)][int(self.col)] == 4 or self.dead:
            self.setTarget()
        self.setDir()
        self.move()

        if self.attacked:
            self.attackedCount += 1

        if self.attacked and not self.dead:
            self.ghostSpeed = 1/8

        if self.attackedCount == self.attackedTimer and self.attacked:
            if not self.dead:
                self.ghostSpeed = 1/4
                self.row = math.floor(self.row)
                self.col = math.floor(self.col)

            self.attackedCount = 0
            self.attacked = False
            self.setTarget()

        if self.dead and game_board[self.row][self.col] == 4:
            self.deathCount += 1
            self.attacked = False
            if self.deathCount == self.deathTimer:
                self.deathCount = 0
                self.dead = False
                self.ghostSpeed = 1/4

    def draw(self): # Ghosts states: Alive, Attacked, Dead Attributes: Color, Direction, Location
        ghostImage = pygame.image.load(element_path + "tile152.png")
        currentDir = ((self.dir + 3) % 4) * 2
        if self.changeFeetCount == self.changeFeetDelay:
            self.changeFeetCount = 0
            currentDir += 1
        self.changeFeetCount += 1
        if self.dead:
            tileNum = 152 + currentDir
            ghostImage = pygame.image.load(element_path + "tile" + str(tileNum) + ".png")
        elif self.attacked:
            if self.attackedTimer - self.attackedCount < self.attackedTimer//3:
                if (self.attackedTimer - self.attackedCount) % 31 < 26:
                    ghostImage = pygame.image.load(element_path + "tile0" + str(70 + (currentDir - (((self.dir + 3) % 4) * 2))) + ".png")
                else:
                    ghostImage = pygame.image.load(element_path + "tile0" + str(72 + (currentDir - (((self.dir + 3) % 4) * 2))) + ".png")
            else:
                ghostImage = pygame.image.load(element_path + "tile0" + str(72 + (currentDir - (((self.dir + 3) % 4) * 2))) + ".png")
        else:
            if self.color == "blue":
                tileNum = 136 + currentDir
                ghostImage = pygame.image.load(element_path + "tile" + str(tileNum) + ".png")
            elif self.color == "pink":
                tileNum = 128 + currentDir
                ghostImage = pygame.image.load(element_path + "tile" + str(tileNum) + ".png")
            elif self.color == "orange":
                tileNum = 144 + currentDir
                ghostImage = pygame.image.load(element_path + "tile" + str(tileNum) + ".png")
            elif self.color == "red":
                tileNum = 96 + currentDir
                if tileNum < 100:
                    ghostImage = pygame.image.load(element_path + "tile0" + str(tileNum) + ".png")
                else:
                    ghostImage = pygame.image.load(element_path + "tile" + str(tileNum) + ".png")

        ghostImage = pygame.transform.scale(ghostImage, (int(square * sprite_ratio), int(square * sprite_ratio)))
        screen.blit(ghostImage, (self.col * square + sprite_offset, self.row * square + sprite_offset, square, square))

    def isValidTwo(self, cRow, cCol, dist, visited):
        if cRow < 3 or cRow >= len(game_board) - 5 or cCol < 0 or cCol >= len(game_board[0]) or game_board[cRow][cCol] == 3:
            return False
        elif visited[cRow][cCol] <= dist:
            return False
        return True

    def isValid(self, cRow, cCol):
        if cCol < 0 or cCol > len(game_board[0]) - 1:
            return True
        for ghost in game.ghosts:
            if ghost.color == self.color:
                continue
            if ghost.row == cRow and ghost.col == cCol and not self.dead:
                return False
        if not self.ghostGate.count([cRow, cCol]) == 0:
            if self.dead and self.row < cRow:
                return True
            elif self.row > cRow and not self.dead and not self.attacked and not game.locked_in:
                return True
            else:
                return False
        if game_board[cRow][cCol] == 3:
            return False
        return True

    def setDir(self): #Very inefficient || can easily refactor
        # BFS search -> Not best route but a route none the less
        dirs = [[0, -self.ghostSpeed, 0],
                [1, 0, self.ghostSpeed],
                [2, self.ghostSpeed, 0],
                [3, 0, -self.ghostSpeed]
        ]
        random.shuffle(dirs)
        best = 10000
        bestDir = -1
        for newDir in dirs:
            if self.calcDistance(self.target, [self.row + newDir[1], self.col + newDir[2]]) < best:
                if not (self.lastLoc[0] == self.row + newDir[1] and self.lastLoc[1] == self.col + newDir[2]):
                    if newDir[0] == 0 and self.col % 1.0 == 0:
                        if self.isValid(math.floor(self.row + newDir[1]), int(self.col + newDir[2])):
                            bestDir = newDir[0]
                            best = self.calcDistance(self.target, [self.row + newDir[1], self.col + newDir[2]])
                    elif newDir[0] == 1 and self.row % 1.0 == 0:
                        if self.isValid(int(self.row + newDir[1]), math.ceil(self.col + newDir[2])):
                            bestDir = newDir[0]
                            best = self.calcDistance(self.target, [self.row + newDir[1], self.col + newDir[2]])
                    elif newDir[0] == 2 and self.col % 1.0 == 0:
                        if self.isValid(math.ceil(self.row + newDir[1]), int(self.col + newDir[2])):
                            bestDir = newDir[0]
                            best = self.calcDistance(self.target, [self.row + newDir[1], self.col + newDir[2]])
                    elif newDir[0] == 3 and self.row % 1.0 == 0:
                        if self.isValid(int(self.row + newDir[1]), math.floor(self.col + newDir[2])):
                            bestDir = newDir[0]
                            best = self.calcDistance(self.target, [self.row + newDir[1], self.col + newDir[2]])
        self.dir = bestDir

    def calcDistance(self, a, b):
        dR = a[0] - b[0]
        dC = a[1] - b[1]
        return math.sqrt((dR * dR) + (dC * dC))

    def setTarget(self):
        if game_board[int(self.row)][int(self.col)] == 4 and not self.dead:
            self.target = [self.ghostGate[0][0] - 1, self.ghostGate[0][1] + 1]
            return
        elif game_board[int(self.row)][int(self.col)] == 4 and self.dead:
            self.target = [self.row, self.col]
        elif self.dead:
            self.target = [14, 13]
            return

        # Records the quadrants of each ghost's target
        quads = [0, 0, 0, 0]
        for ghost in game.ghosts:
            # if ghost.target[0] == self.row and ghost.col == self.col:
            #     continue
            if ghost.target[0] <= 15 and ghost.target[1] >= 13:
                quads[0] += 1
            elif ghost.target[0] <= 15 and ghost.target[1] < 13:
                quads[1] += 1
            elif ghost.target[0] > 15 and ghost.target[1] < 13:
                quads[2] += 1
            elif ghost.target[0]> 15 and ghost.target[1] >= 13:
                quads[3] += 1

        # Finds a target that will keep the ghosts dispersed
        while True:
            self.target = [randrange(31), randrange(28)]
            quad = 0
            if self.target[0] <= 15 and self.target[1] >= 13:
                quad = 0
            elif self.target[0] <= 15 and self.target[1] < 13:
                quad = 1
            elif self.target[0] > 15 and self.target[1] < 13:
                quad = 2
            elif self.target[0] > 15 and self.target[1] >= 13:
                quad = 3
            if not game_board[self.target[0]][self.target[1]] == 3 and not game_board[self.target[0]][self.target[1]] == 4:
                break
            elif quads[quad] == 0:
                break

    def move(self):
        # print(self.target)
        self.lastLoc = [self.row, self.col]
        if self.dir == 0:
            self.row -= self.ghostSpeed
        elif self.dir == 1:
            self.col += self.ghostSpeed
        elif self.dir == 2:
            self.row += self.ghostSpeed
        elif self.dir == 3:
            self.col -= self.ghostSpeed

        # Incase they go through the middle tunnel
        self.col = self.col % len(game_board[0])
        if self.col < 0:
            self.col = len(game_board[0]) - 0.5

    def setAttacked(self, isAttacked):
        self.attacked = isAttacked

    def isAttacked(self):
        return self.attacked

    def setDead(self, isDead):
        self.dead = isDead

    def isDead(self):
        return self.dead


game = Game(1, 0)

def start():
    running = True
    clock = pygame.time.Clock()
    game.render()
    i = 0
    while running :
        clock.tick(15)
        for event in pygame.event.get():
            if event.type == pygame.QUIT or game.game_over:
                running = False
                game.recordHighScore()
                break
            elif event.type == pygame.KEYDOWN:
                game.paused = False
                game.started = True
                if event.key in PLAYING_KEYS["up"]:
                    game.pacman.newDir = 0
                elif event.key in PLAYING_KEYS["right"]:
                    game.pacman.newDir = 1
                elif event.key in PLAYING_KEYS["down"]:
                    game.pacman.newDir = 2
                elif event.key in PLAYING_KEYS["left"]:
                    game.pacman.newDir = 3
                elif event.key == pygame.K_q:
                    running = False
                    game.recordHighScore()
        print(f'{game.getStatePacman()} {game.pacman.newDir} {game.pacman.dir}')
        game.update()
        i += 1

if (__name__=="__main__"):
    start()