import numpy as np
import math

NUM_COLUMNS = 7
COLUMN_HEIGHT = 6
FOUR = 4
WINDOW_LENGTH = 4
PLAYER_1 = 1
PLAYER_2 = -1
MAX_DEPTH = 5
ITERMAX = 2501

ENDC = '\033[0m'
BOLD = '\033[1m'
BOARD_LINE = '\033[94m' + BOLD
FIRST_PLAYER = '\033[93m' + BOLD
SECOND_PLAYER = '\033[91m' + BOLD


class MCTS_Node:
    def __init__(self, board, player, move=None, parent=None):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.player = player
        self.unexploredMoves = board.valid_moves()
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.difficulty = 2

    def selection(self):
        res = lambda x: x.wins / x.visits + np.sqrt(2 * np.log(self.visits) / x
                                                    .visits)
        return sorted(self.childNodes, key=res)[-1]

    def expand(self, move, board):
        child = MCTS_Node(board, self.player, move=move, parent=self)
        self.unexploredMoves.remove(move)
        self.childNodes.append(child)
        return child

    def update(self, result):
        self.wins += result
        self.visits += 1


class Board():
    def __init__(self):
        self.board = np.zeros((NUM_COLUMNS, COLUMN_HEIGHT), dtype=np.byte)

    def refresh_board(self):
        """Re-initialized board"""
        self.board = np.zeros((NUM_COLUMNS, COLUMN_HEIGHT), dtype=np.byte)

    def copy(self):
        """Creates a copy of the current board"""
        b = Board()
        b.board = self.board.copy()
        return b

    def print_board(self):
        """Pretty prints current board to terminal"""
        to_print = np.rot90(self.board)
        output = "\n"
        for row_index in range(COLUMN_HEIGHT):
            output += BOARD_LINE + "\t|" + ENDC
            for col_index in range(NUM_COLUMNS):
                char = "  "
                if to_print[row_index][col_index] == 1:
                    char = FIRST_PLAYER + "⬤ " + ENDC
                elif to_print[row_index][col_index] == -1:
                    char = SECOND_PLAYER + "⬤ " + ENDC
                output += f"{char}" + BOARD_LINE + "|" + ENDC
            output += "\n"

        output += BOARD_LINE + "\t ⓵  ⓶  ⓷  ⓸  ⓹  ⓺  ⓻\n" + ENDC
        output += "-------------------------------------"
        print(output)

    def play(self, column, player):
        """Updates `board` as `player` drops a disc in `column`"""
        (index, ) = next(
            (i for i, v in np.ndenumerate(self.board[column]) if v == 0))
        self.board[column, index] = player

    def still_valid_moves(self):
        """Returns columns where a disc may be played only if nobody has already won"""
        if self.four_in_a_row(PLAYER_1) or self.four_in_a_row(PLAYER_2):
            return []
        return [
            n for n in range(NUM_COLUMNS)
            if self.board[n, COLUMN_HEIGHT - 1] == 0
        ]

    def valid_moves(self):
        """Returns columns where a disc may be played"""
        return [
            n for n in range(NUM_COLUMNS)
            if self.board[n, COLUMN_HEIGHT - 1] == 0
        ]

    def four_in_a_row(self, player):
        """Checks if `player` has a 4-piece line"""
        return (any(
            all(self.board[c, r] == player) for c in range(NUM_COLUMNS)
            for r in (list(range(n, n + FOUR))
                      for n in range(COLUMN_HEIGHT - FOUR + 1))) or any(
                          all(self.board[c, r] == player)
                          for r in range(COLUMN_HEIGHT)
                          for c in (list(range(n, n + FOUR))
                                    for n in range(NUM_COLUMNS - FOUR + 1)))
                or any(
                    np.all(self.board[diag] == player)
                    for diag in ((range(ro, ro + FOUR), range(co, co + FOUR))
                                 for ro in range(0, NUM_COLUMNS - FOUR + 1)
                                 for co in range(0, COLUMN_HEIGHT - FOUR + 1)))
                or any(
                    np.all(self.board[diag] == player)
                    for diag in ((range(ro, ro + FOUR),
                                  range(co + FOUR - 1, co - 1, -1))
                                 for ro in range(0, NUM_COLUMNS - FOUR + 1)
                                 for co in range(0, COLUMN_HEIGHT - FOUR +
                                                 1))))

    def is_leaf_node(self):
        """Checks if current node is terminal"""
        return self.four_in_a_row(PLAYER_1) or self.four_in_a_row(
            PLAYER_2) or len(self.valid_moves()) == 0

    def assign_leaf_node(self):
        """If current node is terminal return the winner or 0 for draw. Otherwise returns None"""
        if self.four_in_a_row(PLAYER_1):
            return 1
        if self.four_in_a_row(PLAYER_2):
            return -1
        if len(self.valid_moves()) == 0:
            return 0
        else:
            return None

    def is_winner(self, player):
        """Checks if player has won in the current node"""
        if self.four_in_a_row(player):
            return 1
        if self.four_in_a_row(-player):
            return 0
        if len(self.valid_moves()) == 0:
            return 0.5
        else:
            return None

    def get_board_score(self, board, player_n):
        """Elaborates an heuristic score for the current node"""
        score = 0
        # Center column positions are more valuable and should be scored more
        center_array = [int(i) for i in list(board[NUM_COLUMNS // 2, :])]
        center_count = center_array.count(player_n)
        score += center_count * 3

        # Checks horizontal lines
        for c in range(COLUMN_HEIGHT):
            row_array = [int(i) for i in list(board[:, c])]
            for c in range(NUM_COLUMNS - 3):
                window = row_array[c:c + WINDOW_LENGTH]
                score += heuristic(window, player_n)

        # Check vertical lines
        for r in range(NUM_COLUMNS):
            col_array = [int(i) for i in list(board[r, :])]
            for r in range(COLUMN_HEIGHT - 3):
                window = col_array[r:r + WINDOW_LENGTH]
                score += heuristic(window, player_n)

        # Checks Diagonal lines
        for r in range(COLUMN_HEIGHT - 3):
            for c in range(NUM_COLUMNS - 3):
                window = [board[c + i][r + i] for i in range(WINDOW_LENGTH)]
                score += heuristic(window, player_n)

        for r in range(COLUMN_HEIGHT - 3):
            for c in range(NUM_COLUMNS - 3):
                window = [
                    board[c + 3 - i][r + i] for i in range(WINDOW_LENGTH)
                ]
                score += heuristic(window, player_n)

        return score

    def minimax(self, board, depth, alpha, beta, maximizingPlayer, player_n):
        """The Minimax algorith with alpha-beta pruning"""
        valid_moves = self.valid_moves()
        np.random.shuffle(valid_moves)
        winner = self.assign_leaf_node()

        if depth == 0 or winner is not None:
            if depth == 0:
                return None, self.get_board_score(board, player_n)
            else:
                if winner == PLAYER_2:
                    return None, 999999999
                elif winner == PLAYER_1:
                    return None, -999999999
                else:  #draw
                    return None, 0

        if maximizingPlayer:
            value = -math.inf
            col = np.random.choice(valid_moves)
            for move in valid_moves:
                board_copy = self.copy()
                board_copy.play(move, player_n)
                new_score = board_copy.minimax(board_copy.board, depth - 1,
                                               alpha, beta, False,
                                               -player_n)[1]
                if new_score > value:
                    value = new_score
                    col = move
                alpha = max(alpha, value)
                if alpha >= beta:
                    break

            return col, value

        else:  #Minimizing player
            value = math.inf
            col = np.random.choice(valid_moves)
            for move in valid_moves:
                board_copy = self.copy()
                board_copy.play(move, player_n)
                new_score = board_copy.minimax(board_copy.board, depth - 1,
                                               alpha, beta, True, -player_n)[1]
                if new_score < value:
                    value = new_score
                    col = move
                beta = max(beta, value)
                if alpha >= beta:
                    break

            return col, value

    def MCTS(self, currentBoard, itermax, player_n, currentNode=None):
        """The Monte Carlo Tree Search algorithm"""
        p_n = -player_n
        rootnode = MCTS_Node(currentBoard, player_n)
        if currentNode is not None:
            rootnode = currentNode

        for i in range(itermax):
            node = rootnode
            board = currentBoard.copy()

            # Selection phase
            while node.unexploredMoves == [] and node.childNodes != []:
                node = node.selection()
                board.play(node.move, p_n)
                p_n *= -1

            p_n = player_n
            # Expansion phase
            if node.unexploredMoves != []:
                m = np.random.choice(node.unexploredMoves)
                board.play(m, p_n)
                p_n *= -1
                node = node.expand(m, board)

            p_n = -player_n
            # Rollout phase
            while board.still_valid_moves():
                board.play(np.random.choice(board.valid_moves()), p_n)
                p_n *= -1

            p_n = player_n
            # Backpropagation phase
            while node is not None:
                node.update(board.is_winner(p_n))
                p_n *= -1
                node = node.parent

        foo = lambda x: x.wins / x.visits
        sortedChildNodes = sorted(rootnode.childNodes, key=foo)[::-1]
        return sortedChildNodes[0].move

    def pick_choice(self):
        """Picks a input from the user to choose the column"""
        print("Choose a column")
        try:
            choice = int(input()) - 1
            if choice not in self.valid_moves():
                print('\033[91m' + BOLD + "\nNOT A VALID MOVE!\n" + ENDC)
                return self.pick_choice()
            else:
                return choice
        except:
            print('\033[91m' + BOLD + "\nINSERT A NUMERIC VALUE!\n" + ENDC)
            return self.pick_choice()

    def choose_AI_move(self, opponent, player_n):
        """Picks the selected AI to play against"""
        if opponent == "MASSIMINO MINIMONI":  # -> Min Max con alpha-beta pruning
            depth = MAX_DEPTH
            if self.difficulty == 1:
                depth -= 1
            elif self.difficulty == 3:
                depth += 1
            col = self.minimax(self.board, depth, -math.inf, math.inf, True,
                               player_n)[0]
            return col
        elif opponent == "CARLA MONTE":  # -> Montecarlo Tree Search
            maxiter = ITERMAX
            if self.difficulty == 1:
                maxiter -= 1000
            elif self.difficulty == 3:
                maxiter += 1000
            return self.MCTS(self, maxiter, player_n)

    def play_human_vs_human(self):
        """Plays an Human vs Human match"""
        first_bool = True
        while not self.is_leaf_node():
            if first_bool:
                self.print_board()
                print(BOLD + "Player1 " + ENDC + "it's your turn!")
                col = self.pick_choice()
                self.play(col, 1)
                first_bool = not first_bool
            else:
                self.print_board()
                print(BOLD + "Player2 " + ENDC + "it's your turn!")
                col = self.pick_choice()
                self.play(col, -1)
                first_bool = not first_bool

        self.print_board()
        if self.four_in_a_row(PLAYER_1):
            print(BOLD + "PLAYER1 WINS!" + ENDC)
        elif self.four_in_a_row(PLAYER_2):
            print(BOLD + "PLAYER2 WINS!" + ENDC)
        else:
            print(BOLD + "Game ended in draw!" + ENDC)

        self.new_game()

    def play_human_vs_AI(self, opponent):
        """Plays an Human vs AI match"""
        self.difficulty = choose_difficulty()

        first_bool = True
        while not self.is_leaf_node():
            if first_bool:
                self.print_board()
                print(BOLD + "HUMAN " + ENDC + "it's your turn!")
                col = self.pick_choice()
                self.play(col, 1)
                first_bool = not first_bool
            else:
                self.print_board()
                print(BOLD + opponent + ENDC + " is thinking...")
                col = self.choose_AI_move(opponent, PLAYER_2)
                self.play(col, -1)
                first_bool = not first_bool

        self.print_board()
        if self.four_in_a_row(PLAYER_1):
            print(BOLD + "HUMAN WINS!" + ENDC)
        elif self.four_in_a_row(PLAYER_2):
            print(BOLD + opponent + " WINS!" + ENDC)
        else:
            print(BOLD + "Game ended in draw!" + ENDC)

        self.new_game()

    def choose_AI(self):
        """Picks the selected AI"""
        print("\nChoose your opponent:")
        print(BOLD + " - 1" + ENDC + ": MASSIMINO MINIMONI")
        print(BOLD + " - 2" + ENDC + ": CARLA MONTE")
        print(BOLD + " - 3" + ENDC + ": Back\n")

        choice = input()
        if choice not in ["1", "2", "3", "4"]:
            self.choose_AI()
        elif choice == "1":
            self.play_human_vs_AI("MASSIMINO MINIMONI")
        elif choice == "2":
            self.play_human_vs_AI("CARLA MONTE")
        elif choice == "3":
            self.new_game()

    def new_game(self):
        """Starts a new game"""
        self.refresh_board()
        print("\nSelect one option:")
        print(BOLD + " - 1" + ENDC + ": Play Human vs Human")
        print(BOLD + " - 2" + ENDC + ": Play Human vs AI")
        print(BOLD + " - 3" + ENDC + ": Play AI vs AI")
        print(BOLD + " - 4" + ENDC + ": Exit\n")

        choice = input()
        if choice not in ["1", "2", "3", "4"]:
            self.start_game()
        elif choice == "1":
            self.play_human_vs_human()
        elif choice == "2":
            self.choose_AI()
        elif choice == "3":
            print("\n" + BOLD + "COMING IN FUTURE UPDATES!" + ENDC)
            self.new_game()
        elif choice == "4":
            exit()

    def start_game(self):
        """Application start"""
        print(BOARD_LINE + "\nWELCOME TO CONNECT4!\n" + ENDC)
        self.new_game()


def heuristic(window, player_n):
    """Checks how many of the same stone there are in a four-slot window and tries to assign a score"""
    score = 0
    opponent = -player_n

    if window.count(player_n) == 3 and window.count(0) == 1:
        score += 10
    elif window.count(player_n) == 2 and window.count(0) == 2:
        score += 4

    if window.count(opponent) == 3 and window.count(0) == 1:
        score -= 10
    elif window.count(opponent) == 2 and window.count(0) == 2:
        score -= 4

    return score


def choose_difficulty():
    """Choses AI difficulty level"""
    print(
        "\nChoose Difficulty:\n(A harder difficulty will increase AI's thinking time)"
    )
    print(BOLD + " - 1" + ENDC + ": Easy")
    print(BOLD + " - 2" + ENDC + ": Medium")
    print(BOLD + " - 3" + ENDC + ": Hard\n")

    try:
        choice = int(input())
        if choice not in [1, 2, 3]:
            print('\033[91m' + BOLD + "\nNOT A VALID DIFFICULTY!\n" + ENDC)
            return choose_difficulty()
        else:
            return choice
    except:
        print('\033[91m' + BOLD + "\nINSERT A NUMERIC VALUE!\n" + ENDC)
        return choose_difficulty()


if __name__ == '__main__':
    b = Board()
    b.start_game()
