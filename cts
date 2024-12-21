import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.graph_objects as go

external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div([
        html.H1("Stock Visualization Dashboard", className="text-center my-4"),
        html.Div([
            html.Label("Please enter the stock name", className="font-weight-bold"),
            dcc.Input(id='stock-input', value='AAPL', type='text', className="form-control"),
            html.Button(id='submit-button', n_clicks=0, children='Submit', className="btn btn-primary mt-2")
        ], className="col-md-6 offset-md-3"),
    ], className="container"),
    html.Div(id='graph-output', className="container mt-4"),
])

@app.callback(
    Output('graph-output', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State('stock-input', 'value')]
)
def update_graph(n_clicks, stock_name):
    if n_clicks > 0:
        try:
            # Fixed date range from 2015 to 2024
            df = yf.download(stock_name, start='2015-01-01', end='2024-12-31')
            if df.empty:
                return html.Div("Error retrieving stock data or no data available for this stock symbol.", className="alert alert-danger")

            # Stock details
            stock_info = yf.Ticker(stock_name).info
            stock_details = html.Div([
                html.H3(f"Stock Details: {stock_info.get('shortName', 'N/A')}", className="text-center"),
                html.Div([
                    html.Div([
                        html.P(f"Sector: {stock_info.get('sector', 'N/A')}"),
                        html.P(f"Industry: {stock_info.get('industry', 'N/A')}"),
                        html.P(f"Market Cap: ${stock_info.get('marketCap', 'N/A'):,}"),
                    ], className="col-md-4"),
                    html.Div([
                        html.P(f"Previous Close: ${stock_info.get('previousClose', 'N/A'):.2f}"),
                        html.P(f"Open: ${stock_info.get('open', 'N/A'):.2f}"),
                        html.P(f"Volume: {stock_info.get('volume', 'N/A'):,}")
                    ], className="col-md-4"),
                    html.Div([
                        html.P(f"Day's Range: {stock_info.get('dayLow', 'N/A')} - {stock_info.get('dayHigh', 'N/A')}"),
                        html.P(f"52 Week Range: {stock_info.get('fiftyTwoWeekLow', 'N/A')} - {stock_info.get('fiftyTwoWeekHigh', 'N/A')}"),
                        html.P(f"Avg Volume: {stock_info.get('averageVolume', 'N/A'):,}")
                    ], className="col-md-4"),
                ], className="row"),
            ], className="mb-4 card p-3")

            # Calculate moving averages
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()

            # Determine buy/sell signal
            if df['SMA_50'].iloc[-1] > df['SMA_200'].iloc[-1]:
                signal = "Buy"
                signal_class = "alert alert-success"
            else:
                signal = "Sell"
                signal_class = "alert alert-danger"

            signal_details = html.Div(f"Recommendation: {signal}", className=f"text-center mt-3 {signal_class}")

            # Closing Price vs Time
            close_fig = go.Figure()
            close_fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close', line=dict(color='blue')))
            close_fig.update_layout(title=f'{stock_name} Closing Prices', xaxis_title='Date', yaxis_title='Close Price')

            # Volume vs Time
            volume_fig = go.Figure()
            volume_fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='green'))
            volume_fig.update_layout(title=f'{stock_name} Volume', xaxis_title='Date', yaxis_title='Volume')

            # Candlestick Chart
            candlestick_fig = go.Figure(data=[go.Candlestick(x=df.index,
                                                             open=df['Open'],
                                                             high=df['High'],
                                                             low=df['Low'],
                                                             close=df['Close'],
                                                             name='Candlestick')])
            candlestick_fig.update_layout(title=f'{stock_name} Candlestick Chart', xaxis_title='Date', yaxis_title='Price')

            # Prediction vs Original
            df = df.dropna().copy()  # Ensure we are working with a copy of the DataFrame
            df['Date'] = df.index.map(pd.Timestamp.toordinal)
            X = df[['Date']]
            y = df['Close']
            
            model = LinearRegression()
            model.fit(X, y)
            
            future_dates = pd.date_range(start=df.index[-1], periods=365).to_pydatetime()
            future_dates_ord = np.array([date.toordinal() for date in future_dates]).reshape(-1, 1)
            future_dates_df = pd.DataFrame(future_dates_ord, columns=['Date'])  # Ensure column name is consistent
            predictions = model.predict(future_dates_df)
            
            prediction_fig = go.Figure()
            prediction_fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Original', line=dict(color='blue')))
            prediction_fig.add_trace(go.Scatter(x=future_dates, y=predictions, mode='lines', name='Prediction', line=dict(color='red')))
            prediction_fig.update_layout(title=f'{stock_name} Prediction vs Original', xaxis_title='Date', yaxis_title='Close Price')

            return html.Div([
                stock_details,
                dcc.Graph(figure=close_fig, config={'displayModeBar': False}),
                dcc.Graph(figure=volume_fig, config={'displayModeBar': False}),
                dcc.Graph(figure=candlestick_fig, config={'displayModeBar': False}),
                dcc.Graph(figure=prediction_fig, config={'displayModeBar': False}),
                signal_details
            ])
        except Exception as e:
            return html.Div(f"Error retrieving stock data: {e}", className="alert alert-danger")
    return html.Div()

if __name__ == '__main__':
    app.run_server(debug=True, port=8052)

import tkinter as tk
from tkinter import messagebox

class TicTacToe:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic Tac Toe")
        self.current_player = "X"
        self.board = [[" " for _ in range(3)] for _ in range(3)]
        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        self.create_buttons()

    def create_buttons(self):
        for row in range(3):
            for col in range(3):
                button = tk.Button(self.root, text=" ", font=('normal', 40), width=5, height=2,
                                   command=lambda r=row, c=col: self.on_button_click(r, c))
                button.grid(row=row, column=col)
                self.buttons[row][col] = button

    def on_button_click(self, row, col):
        if self.board[row][col] == " ":
            self.board[row][col] = self.current_player
            self.buttons[row][col].config(text=self.current_player)
            if self.check_winner():
                messagebox.showinfo("Tic Tac Toe", f"Player {self.current_player} wins!")
                self.reset_board()
            elif self.is_board_full():
                messagebox.showinfo("Tic Tac Toe", "It's a tie!")
                self.reset_board()
            else:
                self.current_player = "O" if self.current_player == "X" else "X"

    def check_winner(self):
        for row in self.board:
            if row[0] == row[1] == row[2] and row[0] != " ":
                return True

        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] and self.board[0][col] != " ":
                return True

        if self.board[0][0] == self.board[1][1] == self.board[2][2] and self.board[0][0] != " ":
            return True
        if self.board[0][2] == self.board[1][1] == self.board[2][0] and self.board[0][2] != " ":
            return True

        return False

    def is_board_full(self):
        for row in self.board:
            if " " in row:
                return False
        return True

    def reset_board(self):
        self.board = [[" " for _ in range(3)] for _ in range(3)]
        for row in range(3):
            for col in range(3):
                self.buttons[row][col].config(text=" ")
        self.current_player = "X"

if __name__ == "__main__":
    root = tk.Tk()
    game = TicTacToe(root)
    root.mainloop()

import pygame
import time
import random

pygame.init()

# Colors
white = (255, 255, 255)
yellow = (255, 255, 102)
black = (0, 0, 0)
red = (213, 50, 80)
green = (0, 255, 0)
blue = (50, 153, 213)

# Display dimensions
dis_width = 800
dis_height = 600

# Set up display
dis = pygame.display.set_mode((dis_width, dis_height))
pygame.display.set_caption('Snake Game')

# Game clock
clock = pygame.time.Clock()
snake_block = 10
snake_speed = 15

# Fonts
font_style = pygame.font.SysFont(None, 50)
score_font = pygame.font.SysFont(None, 35)

# Functions to display score and message
def your_score(score):
    value = score_font.render("Your Score: " + str(score), True, yellow)
    dis.blit(value, [0, 0])

def message(msg, color):
    mesg = font_style.render(msg, True, color)
    dis.blit(mesg, [dis_width / 6, dis_height / 3])

# Function to draw the snake
def our_snake(snake_block, snake_list):
    for x in snake_list:
        pygame.draw.rect(dis, black, [x[0], x[1], snake_block, snake_block])

# Main game loop
def gameLoop():
    game_over = False
    game_close = False

    x1 = dis_width / 2
    y1 = dis_height / 2

    x1_change = 0
    y1_change = 0

    snake_List = []
    Length_of_snake = 1

    foodx = round(random.randrange(0, dis_width - snake_block) / 10.0) * 10.0
    foody = round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0

    while not game_over:

        while game_close == True:
            dis.fill(blue)
            message("You Lost! Press Q-Quit or C-Play Again", red)
            your_score(Length_of_snake - 1)
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        game_over = True
                        game_close = False
                    if event.key == pygame.K_c:
                        gameLoop()
                if event.type == pygame.QUIT:
                    game_over = True
                    game_close = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    x1_change = -snake_block
                    y1_change = 0
                elif event.key == pygame.K_RIGHT:
                    x1_change = snake_block
                    y1_change = 0
                elif event.key == pygame.K_UP:
                    y1_change = -snake_block
                    x1_change = 0
                elif event.key == pygame.K_DOWN:
                    y1_change = snake_block
                    x1_change = 0

        if x1 >= dis_width or x1 < 0 or y1 >= dis_height or y1 < 0:
            game_close = True
        x1 += x1_change
        y1 += y1_change
        dis.fill(white)
        pygame.draw.rect(dis, green, [foodx, foody, snake_block, snake_block])
        snake_Head = []
        snake_Head.append(x1)
        snake_Head.append(y1)
        snake_List.append(snake_Head)
        if len(snake_List) > Length_of_snake:
            del snake_List[0]

        for x in snake_List[:-1]:
            if x == snake_Head:
                game_close = True

        our_snake(snake_block, snake_List)
        your_score(Length_of_snake - 1)

        pygame.display.update()

        if x1 == foodx and y1 == foody:
            foodx = round(random.randrange(0, dis_width - snake_block) / 10.0) * 10.0
            foody = round(random.randrange(0, dis_height - snake_block) / 10.0) * 10.0
            Length_of_snake += 1

        clock.tick(snake_speed)

    pygame.quit()
    quit()

gameLoop()

import pygame
import random
import os

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 600, 600
FPS = 60
CELL_SIZE = 40
COLS, ROWS = WIDTH // CELL_SIZE, HEIGHT // CELL_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
PURPLE = (160, 32, 240)

# Set up display
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pac-Man")

# Load images
def load_image(filename):
    if os.path.exists(filename):
        return pygame.transform.scale(pygame.image.load(filename), (CELL_SIZE, CELL_SIZE))
    return None

pacman_img = load_image("pacman.png")
ghost_images = {
    "red": load_image("ghost_red.png"),
    "blue": load_image("ghost_blue.png"),
    "green": load_image("ghost_green.png"),
    "purple": load_image("ghost_purple.png")
}

# Player class
class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.image = pacman_img
        self.rect = pygame.Rect(self.x * CELL_SIZE, self.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        self.score = 0
        self.move_timer = 0

    def move(self, dx, dy):
        if self.move_timer == 0:
            new_x, new_y = self.x + dx, self.y + dy
            if 0 <= new_x < COLS and 0 <= new_y < ROWS:
                if maze[new_y][new_x] != 1:
                    self.x, self.y = new_x, new_y
                    self.rect.topleft = (self.x * CELL_SIZE, self.y * CELL_SIZE)
                    if (self.x, self.y) in food:
                        food.remove((self.x, self.y))
                        self.score += 10
            self.move_timer = 5  # Set a timer to slow down movement
        else:
            self.move_timer -= 1

    def draw(self):
        # Drawing Pac-Man with a different look
        pygame.draw.circle(window, YELLOW, self.rect.center, CELL_SIZE // 2)
        pygame.draw.polygon(window, BLACK, [
            (self.rect.centerx, self.rect.centery),
            (self.rect.centerx + CELL_SIZE // 2, self.rect.centery - CELL_SIZE // 4),
            (self.rect.centerx + CELL_SIZE // 2, self.rect.centery + CELL_SIZE // 4)
        ])

# Ghost class
class Ghost:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.image = ghost_images[color]
        self.color = color
        self.rect = pygame.Rect(self.x * CELL_SIZE, self.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        self.move_timer = 0

    def move(self):
        if self.move_timer == 0:
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            random.shuffle(directions)
            for dx, dy in directions:
                new_x, new_y = self.x + dx, self.y + dy
                if 0 <= new_x < COLS and 0 <= new_y < ROWS:
                    if maze[new_y][new_x] != 1:
                        self.x, self.y = new_x, new_y
                        self.rect.topleft = (self.x * CELL_SIZE, self.y * CELL_SIZE)
                        break
            self.move_timer = 10  # Set a timer to slow down movement
        else:
            self.move_timer -= 1

    def draw(self):
        color_map = {
            "red": RED,
            "blue": BLUE,
            "green": GREEN,
            "purple": PURPLE
        }
        pygame.draw.circle(window, color_map[self.color], self.rect.center, CELL_SIZE // 2)

# Maze layout
maze = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

def draw_walls():
    for y, row in enumerate(maze):
        for x, cell in enumerate(row):
            if cell == 1:
                pygame.draw.rect(window, BLUE, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

# Food pellets
food = {(x, y) for y, row in enumerate(maze) for x, cell in enumerate(row) if cell == 0}

def draw_food():
    for x, y in food:
        pygame.draw.circle(window, YELLOW, (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2), 5)

# Draw grid
def draw_grid():
    for x in range(0, WIDTH, CELL_SIZE):
        pygame.draw.line(window, BLACK, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, CELL_SIZE):
        pygame.draw.line(window, BLACK, (0, y), (WIDTH, y))

# Draw score
def draw_score(score):
    font = pygame.font.SysFont(None, 36)
    score_surf = font.render(f'Score: {score}', True, WHITE)
    window.blit(score_surf, (10, 10))

# Game loop
def main():
    clock = pygame.time.Clock()
    player = Player(1, 1)
    ghosts = [Ghost(13, 1, "red"), Ghost(13, 13, "blue"), Ghost(1, 13, "green"), Ghost(7, 7, "purple")]

    running = True
    while running:
        clock.tick(FPS)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            player.move(-1, 0)
        if keys[pygame.K_RIGHT]:
            player.move(1, 0)
        if keys[pygame.K_UP]:
            player.move(0, -1)
        if keys[pygame.K_DOWN]:
            player.move(0, 1)

        for ghost in ghosts:
            ghost.move()

        window.fill(BLACK)
        draw_walls()
        draw_food()
        player.draw()
        for ghost in ghosts:
            ghost.draw()
        draw_score(player.score)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()

import tkinter as tk
from time import strftime

root = tk.Tk()
root.resizable(0, 0)
root.title("Python Digital Clock")

# Correct the spelling of 'Label' and add color
label = tk.Label(root, font='arial 20', fg='white', bg='green')
label.pack()

root.geometry("500x100")

def Dclock():
    time_string = strftime('%H:%M:%S %p')
    label.config(text=time_string)
    label.after(1000, Dclock)

Dclock()
root.mainloop()

import numpy as np
import matplotlib.pyplot as plt

R = 2  # Major radius
r = 0.2  # Minor radius
i=19
N=1019
theta = np.linspace(0, 2.*np.pi, 100)
phi = np.linspace(0, 2.*np.pi, 100)
theta, phi = np.meshgrid(theta, phi)

a = (R + r*np.cos(theta)) * np.cos(phi)
b = (R + r*np.cos(theta)) * np.sin(phi)
c = r * np.sin(theta)
u0= 4*np.pi*(10**-7)
mmf= i*N
l=(2*np.pi*R*(10**-2))
H= mmf/l
B= u0*500*H
print("Magnetic Flux Density:",B)
print("Magnetic Field Intensity",H)
fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_zlim(-2,2)
ax1.plot_surface(a,b,c, rstride=50, cstride=10, alpha=0.2,color='pink', edgecolor='black')
ax1.view_init(90,0)

ax2 = fig.add_subplot(122, projection='3d')
ax2.set_zlim(-2,2)
ax2.plot_surface(a, b, c, rstride=50, cstride=10,alpha=0.5, color='pink', edgecolor='black')
ax1.quiver(a, b, c, N,N, 0, length=0.003, normalize=True, color='w')
ax2.quiver(a, b, c,N,N, 0, length=0.003, normalize=True, color='w')

ax2.view_init(25,25)
plt.show()

import java.util.ArrayList;
import java.util.Scanner;

class Book {
    private int id;
    private String title;
    private String author;
    private boolean isAvailable;

    public Book(int id, String title, String author) {
        this.id = id;
        this.title = title;
        this.author = author;
        this.isAvailable = true;
    }

    public int getId() {
        return id;
    }

    public String getTitle() {
        return title;
    }

    public String getAuthor() {
        return author;
    }

    public boolean isAvailable() {
        return isAvailable;
    }

    public void setAvailable(boolean isAvailable) {
        this.isAvailable = isAvailable;
    }

    @Override
    public String toString() {
        return "ID: " + id + ", Title: " + title + ", Author: " + author + ", Available: " + isAvailable;
    }
}

class User {
    private int id;
    private String name;
    private ArrayList<Book> borrowedBooks;

    public User(int id, String name) {
        this.id = id;
        this.name = name;
        this.borrowedBooks = new ArrayList<>();
    }

    public int getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public ArrayList<Book> getBorrowedBooks() {
        return borrowedBooks;
    }

    public void borrowBook(Book book) {
        borrowedBooks.add(book);
    }

    public void returnBook(Book book) {
        borrowedBooks.remove(book);
    }

    @Override
    public String toString() {
        return "ID: " + id + ", Name: " + name + ", Borrowed Books: " + borrowedBooks.size();
    }
}

class Library {
    private ArrayList<Book> books;
    private ArrayList<User> users;

    public Library() {
        books = new ArrayList<>();
        users = new ArrayList<>();
    }

    public void addBook(Book book) {
        books.add(book);
    }

    public void addUser(User user) {
        users.add(user);
    }

    public Book findBookById(int id) {
        for (Book book : books) {
            if (book.getId() == id) {
                return book;
            }
        }
        return null;
    }

    public User findUserById(int id) {
        for (User user : users) {
            if (user.getId() == id) {
                return user;
            }
        }
        return null;
    }

    public void listBooks() {
        if (books.isEmpty()) {
            System.out.println("No books available in the library.");
        } else {
            for (Book book : books) {
                System.out.println(book);
            }
        }
    }

    public void listUsers() {
        if (users.isEmpty()) {
            System.out.println("No users registered in the library.");
        } else {
            for (User user : users) {
                System.out.println(user);
            }
        }
    }

    public boolean borrowBook(int userId, int bookId) {
        User user = findUserById(userId);
        Book book = findBookById(bookId);

        if (user == null || book == null) {
            System.out.println("Invalid user or book ID.");
            return false;
        }

        if (!book.isAvailable()) {
            System.out.println("Book is currently unavailable.");
            return false;
        }

        book.setAvailable(false);
        user.borrowBook(book);
        System.out.println("Book borrowed successfully.");
        return true;
    }

    public boolean returnBook(int userId, int bookId) {
        User user = findUserById(userId);
        Book book = findBookById(bookId);

        if (user == null || book == null) {
            System.out.println("Invalid user or book ID.");
            return false;
        }

        if (!user.getBorrowedBooks().contains(book)) {
            System.out.println("User has not borrowed this book.");
            return false;
        }

        book.setAvailable(true);
        user.returnBook(book);
        System.out.println("Book returned successfully.");
        return true;
    }
}

public class LibraryManagementSystem {
    public static void main(String[] args) {
        Library library = new Library();
        Scanner scanner = new Scanner(System.in);

        while (true) {
            System.out.println("\nLibrary Management System");
            System.out.println("1. Add Book");
            System.out.println("2. Add User");
            System.out.println("3. List Books");
            System.out.println("4. List Users");
            System.out.println("5. Borrow Book");
            System.out.println("6. Return Book");
            System.out.println("7. Exit");
            System.out.print("Enter your choice: ");

            int choice = scanner.nextInt();

            switch (choice) {
                case 1:
                    System.out.print("Enter Book ID: ");
                    int bookId = scanner.nextInt();
                    scanner.nextLine();
                    System.out.print("Enter Book Title: ");
                    String title = scanner.nextLine();
                    System.out.print("Enter Book Author: ");
                    String author = scanner.nextLine();
                    library.addBook(new Book(bookId, title, author));
                    System.out.println("Book added successfully.");
                    break;
                case 2:
                    System.out.print("Enter User ID: ");
                    int userId = scanner.nextInt();
                    scanner.nextLine();
                    System.out.print("Enter User Name: ");
                    String name = scanner.nextLine();
                    library.addUser(new User(userId, name));
                    System.out.println("User added successfully.");
                    break;
                case 3:
                    System.out.println("Listing all books:");
                    library.listBooks();
                    break;
                case 4:
                    System.out.println("Listing all users:");
                    library.listUsers();
                    break;
                case 5:
                    System.out.print("Enter User ID: ");
                    int borrowUserId = scanner.nextInt();
                    System.out.print("Enter Book ID: ");
                    int borrowBookId = scanner.nextInt();
                    library.borrowBook(borrowUserId, borrowBookId);
                    break;
                case 6:
                    System.out.print("Enter User ID: ");
                    int returnUserId = scanner.nextInt();
                    System.out.print("Enter Book ID: ");
                    int returnBookId = scanner.nextInt();
                    library.returnBook(returnUserId, returnBookId);
                    break;
                case 7:
                    System.out.println("Exiting the system. Goodbye!");
                    scanner.close();
                    return;
                default:
                    System.out.println("Invalid choice. Please try again.");
            }
        }
    }
}

import java.io.*;
import java.util.*;

class Task {
    String name;
    String description;
    String deadline;
    int priority;

    public Task(String name, String description, String deadline, int priority) {
        this.name = name;
        this.description = description;
        this.deadline = deadline;
        this.priority = priority;
    }

    @Override
    public String toString() {
        return String.format("Task{name='%s', description='%s', deadline='%s', priority=%d}", 
                             name, description, deadline, priority);
    }
}

class TaskManager {
    private List<Task> tasks;
    private static final String FILE_NAME = "tasks.txt";

    public TaskManager() {
        tasks = new ArrayList<>();
        loadTasks();
    }

    public void addTask(Task task) {
        tasks.add(task);
        saveTasks();
    }

    public void removeTask(String taskName) {
        tasks.removeIf(task -> task.name.equals(taskName));
        saveTasks();
    }

    public void updateTask(String taskName, Task updatedTask) {
        for (int i = 0; i < tasks.size(); i++) {
            if (tasks.get(i).name.equals(taskName)) {
                tasks.set(i, updatedTask);
                break;
            }
        }
        saveTasks();
    }

    public List<Task> getAllTasks() {
        return tasks;
    }

    public List<Task> getSortedTasksByPriority() {
        List<Task> sortedTasks = new ArrayList<>(tasks);
        sortedTasks.sort(Comparator.comparingInt(task -> task.priority));
        return sortedTasks;
    }

    private void saveTasks() {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(FILE_NAME))) {
            for (Task task : tasks) {
                writer.write(task.name + ";" + task.description + ";" + task.deadline + ";" + task.priority);
                writer.newLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void loadTasks() {
        try (BufferedReader reader = new BufferedReader(new FileReader(FILE_NAME))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] data = line.split(";");
                if (data.length == 4) {
                    tasks.add(new Task(data[0], data[1], data[2], Integer.parseInt(data[3])));
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

public class TaskManagementApp {
    private static Scanner scanner = new Scanner(System.in);
    private static TaskManager taskManager = new TaskManager();

    public static void main(String[] args) {
        while (true) {
            showMenu();
            int choice = Integer.parseInt(scanner.nextLine());
            switch (choice) {
                case 1:
                    addTask();
                    break;
                case 2:
                    removeTask();
                    break;
                case 3:
                    updateTask();
                    break;
                case 4:
                    displayTasks();
                    break;
                case 5:
                    displaySortedTasks();
                    break;
                case 6:
                    System.out.println("Exiting...");
                    return;
                default:
                    System.out.println("Invalid choice. Please try again.");
            }
        }
    }

    private static void showMenu() {
        System.out.println("\nTask Management System");
        System.out.println("1. Add Task");
        System.out.println("2. Remove Task");
        System.out.println("3. Update Task");
        System.out.println("4. Display All Tasks");
        System.out.println("5. Display Tasks Sorted by Priority");
        System.out.println("6. Exit");
        System.out.print("Choose an option: ");
    }

    private static void addTask() {
        System.out.print("Enter task name: ");
        String name = scanner.nextLine();
        System.out.print("Enter task description: ");
        String description = scanner.nextLine();
        System.out.print("Enter task deadline (YYYY-MM-DD): ");
        String deadline = scanner.nextLine();
        System.out.print("Enter task priority (1-10): ");
        int priority = Integer.parseInt(scanner.nextLine());
        
        Task task = new Task(name, description, deadline, priority);
        taskManager.addTask(task);
        System.out.println("Task added successfully.");
    }

    private static void removeTask() {
        System.out.print("Enter the name of the task to remove: ");
        String name = scanner.nextLine();
        taskManager.removeTask(name);
        System.out.println("Task removed successfully.");
    }

    private static void updateTask() {
        System.out.print("Enter the name of the task to update: ");
        String name = scanner.nextLine();
        System.out.print("Enter new description: ");
        String description = scanner.nextLine();
        System.out.print("Enter new deadline (YYYY-MM-DD): ");
        String deadline = scanner.nextLine();
        System.out.print("Enter new priority (1-10): ");
        int priority = Integer.parseInt(scanner.nextLine());
        
        Task updatedTask = new Task(name, description, deadline, priority);
        taskManager.updateTask(name, updatedTask);
        System.out.println("Task updated successfully.");
    }

    private static void displayTasks() {
        List<Task> tasks = taskManager.getAllTasks();
        if (tasks.isEmpty()) {
            System.out.println("No tasks available.");
        } else {
            tasks.forEach(System.out::println);
        }
    }

    private static void displaySortedTasks() {
        List<Task> tasks = taskManager.getSortedTasksByPriority();
        if (tasks.isEmpty()) {
            System.out.println("No tasks available.");
        } else {
            tasks.forEach(System.out::println);
        }
    }
}
