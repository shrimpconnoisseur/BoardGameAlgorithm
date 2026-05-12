import tkinter as tk
from tkinter import font as tkfont
import math
import time

# constants
BOARD_SIZE = 3
WIN_CONDITION = 3

CELL_PX = 130
PAD = 18

MAX_BOARD_SIZE = 10
MIN_BOARD_SIZE = 3

SEARCH_DEPTH_BY_SIZE = {
  3: None,  # full minimax
  4: 8,
  5: 6,
  10: 4,
}

NEIGHBOR_RADIUS_BY_SIZE = {
  3: 2,
  4: 2,
  5: 2,
  10: 1,
}

# hex
BG_COLOR = "#0f0f14"
PANEL_COLOR = "#16161f"
LINE_COLOR = "#2a2a3d"
X_COLOR = "#e84393"
O_COLOR = "#43c6e8"

HOVER_COLOR = "#1e1e2e"
WIN_CELL_COLOR = "#1a2a1a"
TEXT_COLOR = "#c9c9d9"
DIM_COLOR = "#555570"
ACCENT_COLOR = "#7c6af7"

LOGGER_BG_COLOR = "#0b0b10"
LOGGER_X_COLOR = "#e84393"
LOGGER_O_COLOR = "#43c6e8"
LOGGER_SYS_COLOR = "#7c6af7"
LOGGER_GOOD_COLOR = "#50e882"
LOGGER_PRUNE_COLOR = "#f5a623"

FONT_TITLE = ("Courier New", 15, "bold")
FONT_MARK = ("Courier New", 40, "bold")
FONT_STATUS = ("Courier New", 12, "bold")
FONT_LOGGER = ("Courier New", 10)
FONT_BTN = ("Courier New", 12, "bold")
FONT_SCORE = ("Courier New", 12)

# forced delay between steps (ms)
AI_MOVE_DELAY = 120       # used in human vs AI mode
ALT_AI_MOVE_DELAY = 700     # used in AI vs AI mode

# game logic
class Board:
  EMPTY = ""
  X = "X"
  O = "O"
  
  def __init__(self, size=BOARD_SIZE, win_condition=WIN_CONDITION):
    if not MIN_BOARD_SIZE <= size <= MAX_BOARD_SIZE:
      raise ValueError(f"BOARD_SIZE must be between {MIN_BOARD_SIZE} and {MAX_BOARD_SIZE}")
    if not 3 <= win_condition <= size:
      raise ValueError("WIN_CONDITION must be at least 3 and no larger than BOARD_SIZE")
    self.size = size
    self.win_condition = win_condition
    self.cells = [[self.EMPTY] * size for _ in range(size)]
    self.move_count = 0
    self._lines = self._build_lines()
    
  def copy(self):
    b = Board(self.size, self.win_condition)
    b.cells = [row[:] for row in self.cells]
    b.move_count = self.move_count
    return b
  
  def make_move(self, row, col, player):
    if self.cells[row][col] == self.EMPTY:
      self.cells[row][col] = player
      self.move_count += 1
      return True
    return False
  
  def undo_move(self, row, col):
    self.cells[row][col] = self.EMPTY
    self.move_count -= 1
    
  def available_moves(self):
    return [(r, c)
            for r in range(self.size)
            for c in range(self.size)
            if self.cells[r][c] == self.EMPTY]
    
  def check_winner(self):
    size, win = self.size, self.win_condition
    for line in self._lines:
      marks = [self.cells[r][c] for r, c in line]
      if marks[0] != self.EMPTY and all(m == marks[0] for m in marks):
        return marks[0], line
    
    if self.move_count == size * size:
      return "draw", []
    return None, []
  
  def _build_lines(self):
    size, win = self.size, self.win_condition
    lines = []
    for r in range(size):
      for c in range(size - win + 1):
        lines.append([(r, c+i) for i in range(win)])
    
    for c in range(size):
      for r in range(size - win + 1):
        lines.append([(r+i, c) for i in range(win)])
        
    for r in range(size - win + 1):
      for c in range(size - win + 1):
        lines.append([(r+i, c+i) for i in range(win)])
        lines.append([(r+i, c+win-1-i) for i in range(win)])
        
    return lines

  def line_windows(self):
    return self._lines

  def state_key(self):
    return tuple(tuple(row) for row in self.cells)
  
# minimax with alpha-beta pruning
class Minimax:
  def __init__(self, maximizing_player, minimizing_player, log_callback=None):
    self.MAX = maximizing_player  # "X" / "O"
    self.MIN = minimizing_player
    self.log = log_callback or (lambda *a, **k: None)
    self.nodes_evaluated = 0
    self.prune_count = 0
    self.cache_hits = 0
    self.cache = {}
    self.max_depth = None
    
  def best_move(self, board):
    self.nodes_evaluated = 0
    self.prune_count = 0
    self.cache_hits = 0
    self.cache = {}
    self.max_depth = SEARCH_DEPTH_BY_SIZE.get(board.size, 4)
    best_score = -math.inf
    best_cell = None
    alpha, beta = -math.inf, math.inf
    
    moves = self._ordered_moves(board, self.MAX)
    if not moves:
      return None

    # Log available moves once, up front
    depth_text = "full" if self.max_depth is None else str(self.max_depth)
    self.log("sys", f"MAX={self.MAX}   depth={depth_text}   candidate moves: {moves}")
    
    move_scores = []
    for move in moves:
      board.make_move(*move, self.MAX)
      score = self._minimax(board, depth=1, is_max=False, alpha=alpha, beta=beta)
      board.undo_move(*move)
      move_scores.append((move, score))
      
      if score > best_score:
        best_score = score
        best_cell = move
      alpha = max(alpha, best_score)

    # Log per-move scores in a single compact block
    for move, score in move_scores:
      marker = " <-- best" if move == best_cell else ""
      self.log("eval", f"  {move} -> {score:+d}{marker}")

    # Log the summary line
    self.log("good", 
             f"Best: {best_cell}  score={best_score:+d}  "
             f"nodes={self.nodes_evaluated}  pruned={self.prune_count}  "
             f"cache={self.cache_hits}")
    return best_cell
  
  def _minimax(self, board, depth, is_max, alpha, beta):
    self.nodes_evaluated += 1
    winner, _ = board.check_winner()
    
    if winner == self.MAX:
      return 1_000_000 - depth
    if winner == self.MIN:
      return depth - 1_000_000
    if winner == "draw":
      return 0

    if self.max_depth is not None and depth >= self.max_depth:
      return self._evaluate(board)

    key = (board.state_key(), is_max, depth)
    if key in self.cache:
      self.cache_hits += 1
      return self.cache[key]
    
    player = self.MAX if is_max else self.MIN
    moves = self._ordered_moves(board, player)
    
    if is_max:
      best = -math.inf
      for move in moves:
        board.make_move(*move, self.MAX)
        val = self._minimax(board, depth+1, False, alpha, beta)
        board.undo_move(*move)
        best = max(best, val)
        alpha = max(alpha, best)
        if beta <= alpha:
          self.prune_count += 1
          break
      self.cache[key] = best
      return best
    else:
      best = math.inf
      for move in moves:
        board.make_move(*move, self.MIN)
        val = self._minimax(board, depth+1, True, alpha, beta)
        board.undo_move(*move)
        best = min(best, val)
        beta = min(beta, best)
        if beta <= alpha:
          self.prune_count += 1
          break
      self.cache[key] = best
      return best

  def _candidate_moves(self, board):
    moves = board.available_moves()
    if board.move_count == 0:
      mid = board.size // 2
      centers = [(mid, mid)]
      if board.size % 2 == 0:
        centers = [(mid - 1, mid - 1), (mid - 1, mid), (mid, mid - 1), (mid, mid)]
      return [move for move in centers if move in moves]

    if board.size <= 5:
      return moves

    radius = NEIGHBOR_RADIUS_BY_SIZE.get(board.size, 1)
    candidates = set()
    for r in range(board.size):
      for c in range(board.size):
        if board.cells[r][c] == Board.EMPTY:
          continue
        for dr in range(-radius, radius + 1):
          for dc in range(-radius, radius + 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < board.size and 0 <= nc < board.size:
              if board.cells[nr][nc] == Board.EMPTY:
                candidates.add((nr, nc))
    return list(candidates) if candidates else moves

  def _ordered_moves(self, board, player):
    moves = self._candidate_moves(board)
    opponent = self.MIN if player == self.MAX else self.MAX
    center = (board.size - 1) / 2

    def move_score(move):
      r, c = move
      board.make_move(r, c, player)
      winner, _ = board.check_winner()
      board.undo_move(r, c)
      if winner == player:
        return 10_000_000

      board.make_move(r, c, opponent)
      opponent_winner, _ = board.check_winner()
      board.undo_move(r, c)
      if opponent_winner == opponent:
        return 9_000_000

      distance = abs(r - center) + abs(c - center)
      return self._move_potential(board, r, c, player) - int(distance * 10)

    return sorted(moves, key=move_score, reverse=True)

  def _move_potential(self, board, row, col, player):
    opponent = self.MIN if player == self.MAX else self.MAX
    score = 0
    for line in board.line_windows():
      if (row, col) not in line:
        continue
      marks = [board.cells[r][c] for r, c in line]
      if opponent not in marks:
        score += 10 ** marks.count(player)
      if player not in marks:
        score += 8 ** marks.count(opponent)
    return score

  def _evaluate(self, board):
    total = 0
    for line in board.line_windows():
      marks = [board.cells[r][c] for r, c in line]
      max_count = marks.count(self.MAX)
      min_count = marks.count(self.MIN)

      if max_count and min_count:
        continue
      if max_count:
        total += self._line_score(max_count, board.win_condition)
      elif min_count:
        total -= self._line_score(min_count, board.win_condition)
      else:
        total += 1
    return total

  def _line_score(self, count, win):
    if count >= win:
      return 1_000_000
    if count == win - 1:
      return 50_000
    if count == win - 2:
      return 2_500
    return 10 ** count
    
# gui (game graphics)
class TicTacToeGame:
  def __init__(self, root):
    self.root = root
    self.root.title("Tic-Tac-Toe - Minimax")
    self.root.configure(bg=BG_COLOR)
    self.root.resizable(False, False)
    
    # game state
    self.board = Board()
    self.mode = "HvA"     # init as "HvA" or "AvA"
    self.human = Board.X
    self.ai_player = Board.O
    self.current = Board.X
    self.game_over = False
    self.win_cells = []
    self.scores = {Board.X: 0, Board.O: 0, "draw": 0}
    self._ai_job = None # same bro
    self.cell_px = self._cell_size()
    self.mark_font = self._mark_font()
    
    self._build_ui()
    self._update_status()

  def _cell_size(self):
    if self.board.size <= 3:
      return 130
    if self.board.size <= 5:
      return 90
    return 48

  def _mark_font(self):
    size = max(18, int(self.cell_px * 0.42))
    return ("Courier New", size, "bold")
    
  # layout
  def _build_ui(self):
    # render top bar
    top = tk.Frame(self.root, bg=BG_COLOR)
    top.pack(fill="x", padx=PAD, pady=(PAD, 0))
    
    tk.Label(top, text=f"TIC-TAC-TOE {BOARD_SIZE}x{BOARD_SIZE}", font=FONT_TITLE,
             bg=BG_COLOR, fg=ACCENT_COLOR).pack(side="left")
    
    # mode toggle
    self.mode_var = tk.StringVar(value="HvA")
    btn_frame = tk.Frame(top, bg=BG_COLOR)
    btn_frame.pack(side="right")
    for text, val in [("Human vs AI", "HvA"), ("AI vs AI", "AvA")]:
      rb = tk.Radiobutton(btn_frame, text=text, 
                          variable=self.mode_var, value=val, 
                          command=self._on_mode_change, bg=BG_COLOR, 
                          fg=TEXT_COLOR, selectcolor=PANEL_COLOR, 
                          activebackground=BG_COLOR, activeforeground=ACCENT_COLOR, 
                          font=FONT_BTN, indicatoron=False, 
                          relief="flat", padx=10, pady=4, 
                          bd=1, highlightthickness=0)
      rb.pack(side="left", padx=2)
      
    # scores
    score_bar = tk.Frame(self.root, bg=PANEL_COLOR)
    score_bar.pack(fill="x", padx=PAD, pady=(8, 0))
    
    self.label_score_x = tk.Label(score_bar, text="X  0", font=FONT_SCORE, bg=PANEL_COLOR, fg=X_COLOR, padx=14, pady=4)
    self.label_score_draw = tk.Label(score_bar, text="Draw  0", font=FONT_SCORE, bg=PANEL_COLOR, fg=DIM_COLOR, padx=14, pady=4)
    self.label_score_o = tk.Label(score_bar, text="O  0", font=FONT_SCORE, bg=PANEL_COLOR, fg=O_COLOR, padx=14, pady=4)
    
    for w in (self.label_score_x, self.label_score_draw, self.label_score_o):
      w.pack(side="left")
      
    # main area (for board)
    main = tk.Frame(self.root, bg=BG_COLOR)
    main.pack(padx=PAD, pady=PAD)
    
    # board
    board_px = self.board.size * self.cell_px
    self.canvas = tk.Canvas(main, width=board_px, height=board_px, bg=PANEL_COLOR, highlightthickness=0)
    
    self.canvas.pack(side="left")
    self.canvas.bind("<Button-1>", self._on_click) # lmb
    self.canvas.bind("<Motion>", self._on_hover)
    self.canvas.bind("<Leave>", lambda e: self._clear_hover())
    self._hover_cell = None
    
    # right panel
    right = tk.Frame(main, bg=BG_COLOR)
    right.pack(side="left", fill="y", padx=(PAD, 0))
    
    # status
    self.label_status = tk.Label(right, text="", font=FONT_STATUS, bg=BG_COLOR, fg=TEXT_COLOR, wraplength=310, justify="left")
    self.label_status.pack(anchor="w", pady=(0, 8))
    
    # logger title
    tk.Label(right, text="Algorithm Logger", font=FONT_BTN, bg=BG_COLOR, fg=DIM_COLOR).pack(anchor="w")
    
    # logger box
    log_frame = tk.Frame(right, bg=LOGGER_BG_COLOR, highlightbackground=LINE_COLOR, highlightthickness=1)
    log_frame.pack(fill="both", expand=True)
    
    self.log_text = tk.Text(log_frame,
                            width=42, height=26,
                            bg=LOGGER_BG_COLOR, fg=TEXT_COLOR,
                            font=FONT_LOGGER,
                            state="disabled",
                            relief="flat", bd=0,
                            padx=8, pady=6,
                            wrap="word",
                            insertbackground=TEXT_COLOR,
                            selectbackground=ACCENT_COLOR)
    self.log_text.pack(side="left", fill="both", expand=True)
    
    sb = tk.Scrollbar(log_frame, command=self.log_text.yview, bg=LOGGER_BG_COLOR, troughcolor=LOGGER_BG_COLOR, activebackground=DIM_COLOR)
    sb.pack(side="right", fill="y")
    self.log_text.config(yscrollcommand=sb.set)
    
    # tag colors
    self.log_text.tag_config("sys", foreground=LOGGER_SYS_COLOR)
    self.log_text.tag_config("eval", foreground=TEXT_COLOR)
    self.log_text.tag_config("good", foreground=LOGGER_GOOD_COLOR)
    self.log_text.tag_config("prune", foreground=LOGGER_PRUNE_COLOR)
    self.log_text.tag_config("x_win", foreground=LOGGER_X_COLOR)
    self.log_text.tag_config("o_win", foreground=LOGGER_O_COLOR)
    self.log_text.tag_config("dim", foreground=DIM_COLOR)
    
    # buttons (apple's newest system because touchscreen was stupid)
    btn_row = tk.Frame(right, bg=BG_COLOR)
    btn_row.pack(fill="x", pady=(8, 0))
    
    self._make_btn(btn_row, "New Game", self._new_game).pack(side="left", padx=(0, 4))
    self._make_btn(btn_row, "Clear Log", self._clear_log).pack(side="left", padx=4)
    self._make_btn(btn_row, "Reset Score", self._reset_scores).pack(side="left", padx=4)
    
    self._draw_board()
    
  def _make_btn(self, parent, text, cmd):
    return tk.Button(parent, text=text, command=cmd,
                     font=FONT_BTN,
                     bg=PANEL_COLOR, fg=TEXT_COLOR,
                     activebackground=ACCENT_COLOR,
                     activeforeground=BG_COLOR,
                     relief="flat", bd=0,
                     padx=10, pady=5,
                     cursor="hand2")
    
  # drawing (the actual board)
  def _draw_board(self):
    self.canvas.delete("all")
    n = self.board.size
    px = self.cell_px
    
    # grid lines
    for i in range(1, n):
      x = i * px
      self.canvas.create_line(x, 4, x, n*px - 4, fill=LINE_COLOR, width=2)
      self.canvas.create_line(4, x, n*px - 4, x, fill=LINE_COLOR, width=2)
      
    # hover highlight
    if self._hover_cell and not self.game_over:
      hr, hc = self._hover_cell
      if self.board.cells[hr][hc] == Board.EMPTY:
        self.canvas.create_rectangle(
          hc*px+2, hr*px+2, (hc+1)*px-2, (hr+1)*px-2, fill=HOVER_COLOR, outline=""
        )
    
    # win highlight
    for r, c in self.win_cells:
      self.canvas.create_rectangle(
        c*px+2, r*px+2, (c+1)*px-2, (r+1)*px-2, fill=WIN_CELL_COLOR, outline=""
      )
      
    # marks
    for r in range(n):
      for c in range(n):
        mark = self.board.cells[r][c]
        if mark:
          cx = c*px + px//2
          cy = r*px + px//2
          clr = X_COLOR if mark == Board.X else O_COLOR
          self.canvas.create_text(cx, cy, text=mark, font=self.mark_font, fill=clr)
  
  # actually logging messages here ok
  def _log(self, tag, msg):
    self.log_text.config(state="normal")
    self.log_text.insert("end", msg + "\n", tag)
    self.log_text.see("end")
    self.log_text.config(state="disabled")
    
  def _clear_log(self):
    self.log_text.config(state="normal")
    self.log_text.delete("1.0", "end")
    self.log_text.config(state="disabled")
    
  # game status logs
  def _update_status(self, msg=None):
    if msg:
      self.label_status.config(text=msg)
      return
    if self.game_over:
      return
    p = self.current
    clr = X_COLOR if p == Board.X else O_COLOR
    if self.mode == "AvA":
      txt = f"Next: {p}   (AI is thinking...)"
    elif p == self.human:
      txt = f"Your turn   ({p})"
    else:
      txt = f"AI is thinking...   ({p})"
    self.label_status.config(text=txt, fg=clr)
    
  def _update_score_labels(self):
    self.label_score_x.config(text=f"X  {self.scores[Board.X]}")
    self.label_score_o.config(text=f"O  {self.scores[Board.O]}")
    self.label_score_draw.config(text=f"Draw  {self.scores['draw']}")
    
  # input handling
  def _cell_from_event(self, event):
    c = event.x // self.cell_px
    r = event.y // self.cell_px
    n = self.board.size
    if 0 <= r < n and 0 <= c < n:
      return r, c
    return None
  
  def _on_hover(self, event):
    cell = self._cell_from_event(event)
    if cell != self._hover_cell:
      self._hover_cell = cell
      self._draw_board()
      
  def _clear_hover(self):
    self._hover_cell = None
    self._draw_board()
    
  def _on_click(self, event):
    if self.game_over:
      return
    if self.mode == "AvA":
      return
    if self.current != self.human:
      return
    cell = self._cell_from_event(event)
    if cell is None:
      return
    r, c = cell
    if self.board.cells[r][c] != Board.EMPTY:
      return
    self._apply_move(r, c, self.human)
    
  # player move
  def _apply_move(self, row, col, player):
    self.board.make_move(row, col, player)
    self._draw_board()
    
    winner, win_line = self.board.check_winner()
    if winner:
      self.win_cells = win_line
      self._draw_board()
      self.game_over = True
      if winner == "draw":
        self.scores["draw"] += 1
        msg = "DRAW"
        self._log("dim", f"\nDRAW")
      else:
        self.scores[winner] += 1
        label = "Human" if (self.mode == "HvA" and winner == self.human) else "AI"
        if self.mode == "AvA":
          label = f"AI ({winner})"
        msg = f"{label} wins!"
        tag = "x_win" if winner == Board.X else "o_win"
        self._log(tag, f"\n{winner} wins!")
      self._update_status(msg)
      self._update_score_labels()
      return
    
    # switch and continue
    self.current = Board.O if player == Board.X else Board.X
    self._update_status()
    
    # trigger AI move if needed
    if not self.game_over:
      if self.mode == "AvA":
        self._ai_job = self.root.after(ALT_AI_MOVE_DELAY, self._ai_turn)
      elif self.current != self.human:
        self._ai_job = self.root.after(AI_MOVE_DELAY, self._ai_turn)
        
  # ai move
  def _ai_turn(self):
    if self.game_over:
      return
    ai = self.current
    opp = Board.O if ai == Board.X else Board.X
    engine = Minimax(ai, opp, log_callback=self._log)
    
    # guard against: no moves available
    # this shouldn't happen, but still prevents crashes
    if not self.board.available_moves():
      return
    
    # python was complaining because `engine.best_move(self.board)` CAN return None, even though it shouldn't.
    move = engine.best_move(self.board)
    if move is None:  # similar guard in best_move
      return          
    r, c = move
    self._apply_move(r, c, ai)
    
  # game management
  def _new_game(self):
    if self._ai_job:
      self.root.after_cancel(self._ai_job)
      self._ai_job = None
    self.board = Board()
    self.current = Board.X
    self.game_over = False
    self.win_cells = []
    self._hover_cell = None
    self._draw_board()
    self._log("sys", "\nNew Game Started")
    self._update_status()
    
    # if ai goes first in AvA or AI is X in HvA
    if self.mode == "AvA":
      self._ai_job = self.root.after(ALT_AI_MOVE_DELAY, self._ai_turn)
    elif self.mode == "HvA" and self.current != self.human:
      self._ai_job = self.root.after(AI_MOVE_DELAY, self._ai_turn)
  
  def _reset_scores(self):
    self.scores = {Board.X: 0, Board.O: 0, "draw": 0}
    self._update_score_labels()
    
  def _on_mode_change(self):
    self.mode = self.mode_var.get()
    self._new_game()
    
# startup
def main():
  root = tk.Tk()
  app = TicTacToeGame(root)
  root.mainloop()
  
if __name__ == "__main__":
  main()
