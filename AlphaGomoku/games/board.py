
# !/usr/bin/python

# Tic-Tac-Toe
# pyGame Demo
# Nathan R. Yergler
# 13 May 2002

# import necessary modules
import pygame
from pygame.locals import *
from common.base_game_spec import BaseGameSpec
import numpy as np

# declare our global variables for the game
XO = "X"  # track whose turn it is; X goes first
grid = [[None, None, None], \
        [None, None, None], \
        [None, None, None]]

winner = None


# declare our support functions

def initBoard(ttt, board_size):
    # initialize the board and return it as a variable
    # ---------------------------------------------------------------
    # ttt : a properly initialized pyGame display variable

    # set up the background surface
    background = pygame.Surface(ttt.get_size())
    background = background.convert()
    background.fill((250, 250, 250))

    # draw the grid lines
    # vertical lines...
    for i in range(board_size):
        pygame.draw.line(background, (0, 0, 0), (i*50, 0), (i*50, 50*board_size), 2)
        # pygame.draw.line(background, (0, 0, 0), (200, 0), (200, 300), 2)

    # horizontal lines...
    for i in range(board_size):
        pygame.draw.line(background, (0, 0, 0), (0, i*50), (50*board_size, i*50), 2)
        # pygame.draw.line(background, (0, 0, 0), (0, 200), (300, 200), 2)

    # return the board
    return background


def drawStatus(board):
    # draw the status (i.e., player turn, etc) at the bottom of the board
    # ---------------------------------------------------------------
    # board : the initialized game board surface where the status will
    #         be drawn

    # gain access to global variables
    global XO, winner

    # determine the status message
    if (winner is None):
        message = XO + "'s turn"
    else:
         message = winner + " won!"

    # render the status message
    font = pygame.font.Font(None, 24)
    text = font.render(message, 1, (10, 10, 10))

    # copy the rendered message onto the board
    # board.fill((250, 250, 250), (0, 300, 300, 25))
    # board.blit(text, (10, 300))


def showBoard(ttt, board):
    # redraw the game board on the display
    # ---------------------------------------------------------------
    # ttt   : the initialized pyGame display
    # board : the game board surface

    drawStatus(board)
    ttt.blit(board, (0, 0))
    pygame.display.flip()


def boardPos(mouseX,mouseY):
    # given a set of coordinates from the mouse, determine which board space
    # (row, column) the user clicked in.
    # ---------------------------------------------------------------
    # mouseX : the X coordinate the user clicked
    # mouseY : the Y coordinate the user clicked
    # determine the row the user clicked
    if (mouseY < 100):
        row = 0
    elif (mouseY < 200):
        row = 1
    else:
        row = 2

    # # determine the column the user clicked
    if (mouseX < 100):
        col = 0
    elif (mouseX < 200):
        col = 1
    else:
        col = 2

    # return the tuple containg the row & column
    return (row, col)


def drawMove(board, boardRow, boardCol, Piece):
    # draw an X or O (Piece) on the board in boardRow, boardCol
    # ---------------------------------------------------------------
    # board     : the game board surface
    # boardRow,
    # boardCol  : the Row & Col in which to draw the piece (0 based)
    # Piece     : X or O

    # determine the center of the square
    centerX = ((boardCol) * 50) + 25
    centerY = ((boardRow) * 50) + 25

    # draw 'O' for player 1 and 'X' for player -1
    if (Piece == 1):
        pygame.draw.circle(board, (0, 0, 0), (centerX, centerY), 22, 2)
    else:
        pygame.draw.line(board, (0, 0, 0), (centerX - 11, centerY - 11), \
                         (centerX + 11, centerY + 11), 2)
        pygame.draw.line(board, (0, 0, 0), (centerX + 11, centerY - 11), \
                         (centerX - 11, centerY + 11), 2)

    # mark the space as used
    # grid[boardRow][boardCol] = Piece


def clickBoard(board,board_state):
    # determine where the user clicked and if the space is not already
    # occupied, draw the appropriate piece there (X or O)
    # ---------------------------------------------------------------
    # board : the game board surface

    global grid, XO

    (mouseX, mouseY) = pygame.mouse.get_pos()
    # parse the board state and populate the board for display.
    # check for -1,1 or 0

    temp = np.argwhere(board_state == 1)
    temp = np.delete(temp,2,1)
    i = 0
    while (i < len(temp)):
        pos = temp[i]
        drawMove(board, pos[0], pos[1], 1)
        i+=1

    temp2 = np.argwhere(board_state == -1)
    temp2 = np.delete(temp2, 2, 1)
    i = 0
    while (i < len(temp2)):
        pos = temp2[i]
        drawMove(board, pos[0], pos[1], -1)
        i+=1
        # (row, col) = boardPos(temp[i])
    # (row, col) = boardPos(mouseX, mouseY)

    # make sure no one's used this space
    # if ((grid[row][col] == "X") or (grid[row][col] == "O")):
    #     # this space is in use
    #     return

    # draw an X or O


    # toggle XO to the other player's move
    # if (XO == "X"):
    #     XO = "O"
    # else:
    #     XO = "X"


def gameWon(board):
    # determine if anyone has won the game
    # ---------------------------------------------------------------
    # board : the game board surface

    global grid, winner

    # check for winning rows
    for row in range(0, 3):
        if ((grid[row][0] == grid[row][1] == grid[row][2]) and \
                    (grid[row][0] is not None)):
            # this row won
            winner = grid[row][0]
            pygame.draw.line(board, (250, 0, 0), (0, (row + 1) * 100 - 50), \
                             (300, (row + 1) * 100 - 50), 2)
            break

    # check for winning columns
    for col in range(0, 3):
        if (grid[0][col] == grid[1][col] == grid[2][col]) and \
                (grid[0][col] is not None):
            # this column won
            winner = grid[0][col]
            pygame.draw.line(board, (250, 0, 0), ((col + 1) * 100 - 50, 0), \
                             ((col + 1) * 100 - 50, 300), 2)
            break

    # check for diagonal winners
    if (grid[0][0] == grid[1][1] == grid[2][2]) and \
            (grid[0][0] is not None):
        # game won diagonally left to right
        winner = grid[0][0]
        pygame.draw.line(board, (250, 0, 0), (50, 50), (250, 250), 2)

    if (grid[0][2] == grid[1][1] == grid[2][0]) and \
            (grid[0][2] is not None):
        # game won diagonally right to left
        winner = grid[0][2]
        pygame.draw.line(board, (250, 0, 0), (250, 50), (50, 250), 2)


# --------------------------------------------------------------------
# initialize pygame and our window
class boardDisp(BaseGameSpec):
    def __init__(self, board_size, board_state):
        pygame.init()
        ttt = pygame.display.set_mode((50 * board_size, 50 * board_size))
        pygame.display.set_caption('Tic-Tac-Toe')
        print(board_state)
        board = initBoard(ttt, board_size)
        running = 1
        start_ticks = pygame.time.get_ticks()
        while (running == 1):
            seconds = (pygame.time.get_ticks() - start_ticks) / 1000  # calculate how many seconds
            if seconds > 4:
                running = 0

            clickBoard(board, board_state)
            showBoard(ttt, board)

if __name__ == '__main__':
    pygame.init()
    board_size = 3
    ttt = pygame.display.set_mode((50*board_size, 50*board_size))
    board_state = np.array([[[0],[0],[-1]],[[1],[0],[-1]],[[1],[1],[-1]]])
    pygame.display.set_caption('Tic-Tac-Toe')

    board = initBoard(ttt, board_size)
    running  = 1
    start_ticks = pygame.time.get_ticks()
    while (running == 1):
        # for event in pygame.event.get():
        #         if event.type is NOEVENT:
        seconds = (pygame.time.get_ticks() - start_ticks) / 1000  # calculate how many seconds
        if seconds > 10:
            running = 0

        clickBoard(board, board_state)
        showBoard(ttt, board)
            # elif event.type is MOUSEBUTTONDOWN:
                    # the user clicked; place an X or O



                # check for a winner
        # gameWon(board)

           # update the display


     # starter tick


