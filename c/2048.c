/*
 ============================================================================
 Name        : 2048.c
 Author      : Maurits van der Schee
 Description : Console version of the game "2048" for GNU/Linux
 ============================================================================
 */

#define _XOPEN_SOURCE 500  // for: usleep
#include <assert.h>
#include <signal.h>   // defines: signal, SIGINT
#include <stdbool.h>  // defines: true, false
#include <stdint.h>   // defines: uint8_t, uint32_t
#include <stdio.h>    // defines: printf, puts, getchar
#include <stdlib.h>   // defines: EXIT_SUCCESS
#include <string.h>   // defines: strcmp
#include <time.h>     // defines: time

#if !(defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__))
    #include <termios.h>  // defines: termios, TCSANOW, ICANON, ECHO
    #include <unistd.h>   // defines: STDIN_FILENO, usleep
#endif

#define SIZE 4
#define MAX_DEPTH 9999
#define NUM_ROLLOUTS 500
#define NUM_GAMES 100

enum move_type { UP, DOWN, LEFT, RIGHT };
enum methods { MAX, MERGE_SCORE, SUM, SUM_WEIGHTED };
#define METHOD 2

// this function receives 2 pointers (indicated by *) so it can set their values
void getColors(uint8_t value, uint8_t scheme, uint8_t *foreground,
               uint8_t *background)
{
    uint8_t original[] = {8,   255, 1,   255, 2,   255, 3,   255, 4,   255, 5,
                          255, 6,   255, 7,   255, 9,   0,   10,  0,   11,  0,
                          12,  0,   13,  0,   14,  0,   255, 0,   255, 0};
    uint8_t blackwhite[] = {232, 255, 234, 255, 236, 255, 238, 255,
                            240, 255, 242, 255, 244, 255, 246, 0,
                            248, 0,   249, 0,   250, 0,   251, 0,
                            252, 0,   253, 0,   254, 0,   255, 0};
    uint8_t bluered[] = {235, 255, 63,  255, 57,  255, 93,  255, 129, 255, 165,
                         255, 201, 255, 200, 255, 199, 255, 198, 255, 197, 255,
                         196, 255, 196, 255, 196, 255, 196, 255, 196, 255};
    uint8_t *schemes[] = {original, blackwhite, bluered};
    // modify the 'pointed to' variables (using a * on the left hand of the
    // assignment)
    *foreground = *(schemes[scheme] + (1 + value * 2) % sizeof(original));
    *background = *(schemes[scheme] + (0 + value * 2) % sizeof(original));
    // alternatively we could have returned a struct with two variables
}

uint8_t getDigitCount(uint32_t number)
{
    uint8_t count = 0;
    do {
        number /= 10;
        count += 1;
    } while (number);
    return count;
}

void printBoard(uint8_t board[SIZE][SIZE])
{
    for (int i = 0; i < SIZE; i++) {
        printf("[");
        for (int j = 0; j < SIZE - 1; j++) {
            printf("%d, ", ((uint32_t)1 << board[i][j]));
        }
        printf("%d]\n", ((uint32_t)1 << board[i][SIZE - 1]));
    }
}

void drawBoard(uint8_t board[SIZE][SIZE], uint8_t scheme, uint32_t score)
{
    uint8_t x, y, fg, bg;
    printf("\033[H");  // move cursor to 0,0
    printf("2048.c %17d pts\n\n", score);
    for (y = 0; y < SIZE; y++) {
        for (x = 0; x < SIZE; x++) {
            // send the addresses of the foreground and background variables,
            // so that they can be modified by the getColors function
            getColors(board[x][y], scheme, &fg, &bg);
            printf("\033[38;5;%d;48;5;%dm", fg, bg);  // set color
            printf("       ");
            printf("\033[m");  // reset all modes
        }
        printf("\n");
        for (x = 0; x < SIZE; x++) {
            getColors(board[x][y], scheme, &fg, &bg);
            printf("\033[38;5;%d;48;5;%dm", fg, bg);  // set color
            if (board[x][y] != 0) {
                uint32_t number = 1 << board[x][y];
                uint8_t t = 7 - getDigitCount(number);
                printf("%*s%u%*s", t - t / 2, "", number, t / 2, "");
            } else {
                printf("   ·   ");
            }
            printf("\033[m");  // reset all modes
        }
        printf("\n");
        for (x = 0; x < SIZE; x++) {
            getColors(board[x][y], scheme, &fg, &bg);
            printf("\033[38;5;%d;48;5;%dm", fg, bg);  // set color
            printf("       ");
            printf("\033[m");  // reset all modes
        }
        printf("\n");
    }
    printf("\n");
    printf("        ←,↑,→,↓ or q        \n");
    printf("\033[A");  // one line up
}

uint8_t findTarget(uint8_t array[SIZE], uint8_t x, uint8_t stop)
{
    uint8_t t;
    // if the position is already on the first, don't evaluate
    if (x == 0) {
        return x;
    }
    for (t = x - 1;; t--) {
        if (array[t] != 0) {
            if (array[t] != array[x]) {
                // merge is not possible, take next position
                return t + 1;
            }
            return t;
        } else {
            // we should not slide further, return this one
            if (t == stop) {
                return t;
            }
        }
    }
    // we did not find a target
    return x;
}

bool slideArray(uint8_t array[SIZE], uint32_t *score, uint32_t *num_merges)
{
    bool success = false;
    uint8_t x, t, stop = 0;

    for (x = 0; x < SIZE; x++) {
        if (array[x] != 0) {
            t = findTarget(array, x, stop);
            // if target is not original position, then move or merge
            if (t != x) {
                // if target is zero, this is a move
                if (array[t] == 0) {
                    array[t] = array[x];
                } else if (array[t] == array[x]) {
                    // merge (increase power of two)
                    array[t]++;
                    // increase score
                    *score += (uint32_t)1 << array[t];
                    // set stop to avoid double merge
                    stop = t + 1;
                    // increase number of merges
                    *num_merges += 1;
                }
                array[x] = 0;
                success = true;
            }
        }
    }
    return success;
}

void rotateBoard(uint8_t board[SIZE][SIZE])
{
    uint8_t i, j, n = SIZE;
    uint8_t tmp;
    for (i = 0; i < n / 2; i++) {
        for (j = i; j < n - i - 1; j++) {
            tmp = board[i][j];
            board[i][j] = board[j][n - i - 1];
            board[j][n - i - 1] = board[n - i - 1][n - j - 1];
            board[n - i - 1][n - j - 1] = board[n - j - 1][i];
            board[n - j - 1][i] = tmp;
        }
    }
}

bool moveUp(uint8_t board[SIZE][SIZE], uint32_t *score, uint32_t *num_merges)
{
    bool success = false;
    uint8_t x;
    for (x = 0; x < SIZE; x++) {
        success |= slideArray(board[x], score, num_merges);
    }
    return success;
}

bool moveLeft(uint8_t board[SIZE][SIZE], uint32_t *score,  uint32_t *num_merges)
{
    bool success;
    rotateBoard(board);
    success = moveUp(board, score, num_merges);
    rotateBoard(board);
    rotateBoard(board);
    rotateBoard(board);
    return success;
}

bool moveDown(uint8_t board[SIZE][SIZE], uint32_t *score,  uint32_t *num_merges)
{
    bool success;
    rotateBoard(board);
    rotateBoard(board);
    success = moveUp(board, score, num_merges);
    rotateBoard(board);
    rotateBoard(board);
    return success;
}

bool moveRight(uint8_t board[SIZE][SIZE], uint32_t *score,  uint32_t *num_merges)
{
    bool success;
    rotateBoard(board);
    rotateBoard(board);
    rotateBoard(board);
    success = moveUp(board, score, num_merges);
    rotateBoard(board);
    return success;
}

bool findPairDown(uint8_t board[SIZE][SIZE])
{
    bool success = false;
    uint8_t x, y;
    for (x = 0; x < SIZE; x++) {
        for (y = 0; y < SIZE - 1; y++) {
            if (board[x][y] == board[x][y + 1]) return true;
        }
    }
    return success;
}

uint8_t countEmpty(uint8_t board[SIZE][SIZE])
{
    uint8_t x, y;
    uint8_t count = 0;
    for (x = 0; x < SIZE; x++) {
        for (y = 0; y < SIZE; y++) {
            if (board[x][y] == 0) {
                count++;
            }
        }
    }
    return count;
}

bool gameEnded(uint8_t board[SIZE][SIZE])
{
    bool ended = true;
    if (countEmpty(board) > 0) return false;
    if (findPairDown(board)) return false;
    rotateBoard(board);
    if (findPairDown(board)) ended = false;
    rotateBoard(board);
    rotateBoard(board);
    rotateBoard(board);
    return ended;
}

void addRandom(uint8_t board[SIZE][SIZE])
{
    uint8_t x, y;
    uint8_t r, len = 0;
    uint8_t n, list[SIZE * SIZE][2];

    for (x = 0; x < SIZE; x++) {
        for (y = 0; y < SIZE; y++) {
            if (board[x][y] == 0) {
                list[len][0] = x;
                list[len][1] = y;
                len++;
            }
        }
    }

    if (len > 0) {
        r = rand() % len;
        x = list[r][0];
        y = list[r][1];
        n = (rand() % 10) / 9 + 1;
        board[x][y] = n;
    }
}

bool move(uint8_t board[SIZE][SIZE], enum move_type move, uint32_t *score,  uint32_t *num_merges)
{
    // move should be 0-3
    bool success;
    if (move == UP) {
        success = moveUp(board, score, num_merges);
    } else if (move == DOWN) {
        success = moveDown(board, score, num_merges);
    } else if (move == LEFT) {
        success = moveLeft(board, score, num_merges);
    } else {
        success = moveRight(board, score, num_merges);
    }
    if (success) addRandom(board);
    return success;
}
void initBoard(uint8_t board[SIZE][SIZE])
{
    uint8_t x, y;
    for (x = 0; x < SIZE; x++) {
        for (y = 0; y < SIZE; y++) {
            board[x][y] = 0;
        }
    }
    addRandom(board);
    // addRandom(board);
}



void setBufferedInput(bool enable)
{
    #if !(defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__))
    static bool enabled = true;
    static struct termios old;
    struct termios new;

    if (enable && !enabled) {
        // restore the former settings
        tcsetattr(STDIN_FILENO, TCSANOW, &old);
        // set the new state
        enabled = true;
    } else if (!enable && enabled) {
        // get the terminal settings for standard input
        tcgetattr(STDIN_FILENO, &new);
        // we want to keep the old setting to restore them at the end
        old = new;
        // disable canonical mode (buffered i/o) and local echo
        new.c_lflag &= (~ICANON & ~ECHO);
        // set the new settings immediately
        tcsetattr(STDIN_FILENO, TCSANOW, &new);
        // set the new state
        enabled = false;
    }
    #endif
}

int test()
{
    uint8_t array[SIZE];
    // these are exponents with base 2 (1=2 2=4 3=8)
    // data holds per line: 4x IN, 4x OUT, 1x POINTS
    uint8_t data[] = {0, 0, 0, 1, 1,  0, 0, 0, 0, 0, 0, 1, 1, 2,  0, 0, 0,
                      4, 0, 1, 0, 1,  2, 0, 0, 0, 4, 1, 0, 0, 1,  2, 0, 0,
                      0, 4, 1, 0, 1,  0, 2, 0, 0, 0, 4, 1, 1, 1,  0, 2, 1,
                      0, 0, 4, 1, 0,  1, 1, 2, 1, 0, 0, 4, 1, 1,  0, 1, 2,
                      1, 0, 0, 4, 1,  1, 1, 1, 2, 2, 0, 0, 8, 2,  2, 1, 1,
                      3, 2, 0, 0, 12, 1, 1, 2, 2, 2, 3, 0, 0, 12, 3, 0, 1,
                      1, 3, 2, 0, 0,  4, 2, 0, 1, 1, 2, 2, 0, 0,  4};
    uint8_t *in, *out, *points;
    uint8_t t, tests;
    uint8_t i;
    bool success = true;
    uint32_t score;
    uint32_t num_merges;

    tests = (sizeof(data) / sizeof(data[0])) / (2 * SIZE + 1);
    for (t = 0; t < tests; t++) {
        in = data + t * (2 * SIZE + 1);
        out = in + SIZE;
        points = in + 2 * SIZE;
        for (i = 0; i < SIZE; i++) {
            array[i] = in[i];
        }
        score = 0;
        slideArray(array, &score, &num_merges);
        for (i = 0; i < SIZE; i++) {
            if (array[i] != out[i]) {
                success = false;
            }
        }
        if (score != *points) {
            success = false;
        }
        if (success == false) {
            for (i = 0; i < SIZE; i++) {
                printf("%d ", in[i]);
            }
            printf("=> ");
            for (i = 0; i < SIZE; i++) {
                printf("%d ", array[i]);
            }
            printf("(%d points) expected ", score);
            for (i = 0; i < SIZE; i++) {
                printf("%d ", in[i]);
            }
            printf("=> ");
            for (i = 0; i < SIZE; i++) {
                printf("%d ", out[i]);
            }
            printf("(%d points)\n", *points);
            break;
        }
    }
    if (success) {
        printf("All %u tests executed successfully\n", tests);
    }
    return !success;
}

void signal_callback_handler(int signum)
{
    printf("         TERMINATED         \n");
    setBufferedInput(true);
    // make cursor visible, reset all modes
    printf("\033[?25h\033[m");
    exit(signum);
}

void copy_board(uint8_t dest[SIZE][SIZE], const uint8_t board[SIZE][SIZE])
{
    memcpy(dest, board, SIZE * SIZE * sizeof(uint8_t));
    // for (int i = 0; i < SIZE; i++) {
    //     for (int j = 0; j < SIZE; j++) {
    //         dest[i][j] = board[i][j];
    //     }
    // }
}

uint32_t max_tile(uint8_t board[SIZE][SIZE])
{
    uint32_t max = 0;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            uint32_t val = (uint32_t)1 << board[i][j];
            if (val > max) {
                max = val;
            }
        }
    }
    return max;
}

uint32_t sum_tile(uint8_t board[SIZE][SIZE])
{
    uint32_t sum = 0;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            sum += (uint32_t)1 << board[i][j];
        }
    }
    return sum;
}

uint32_t num_empty_tile(uint8_t board[SIZE][SIZE])
{
    uint32_t n = 0;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (board[i][j] == 0) n++;
        }
    }
    return n;
}

bool board_eq(uint8_t board1[SIZE][SIZE], uint8_t board2[SIZE][SIZE])
{
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (board1[i][j] != board2[i][j]) {
                return false;
            }
        }
    }
    return true;
}

int random_run(uint8_t board[SIZE][SIZE])
{
    uint8_t board_copy[SIZE][SIZE];
    copy_board(board_copy, board);
    assert(board_eq(board, board_copy));
    int i = 0;
    uint32_t _score = 0, _num_merges = 0;
    while (!gameEnded(board_copy)) {
        enum move_type random_move = ((uint32_t)rand()) % 4;
        bool succ = move(board_copy, random_move, &_score, &_num_merges);
        i++;
    }
#if METHOD == 0
    return max_tile(board_copy);
#elif METHOD == 1
    return _score;
#elif METHOD == 2
    return sum_tile(board_copy);
#elif METHOD == 3
    // prefer more zero tiles
    return sum_tile(board_copy) + num_empty_tile(board_copy);
#endif
}

void monte_carlo_iter(uint8_t board[SIZE][SIZE], uint32_t *score, uint32_t * num_merges, int num_iter)
{
    int scores[4] = {0};
    for (int m = 0; m <= 3; m++) {
        uint32_t total_score = 0, tmp_score = 0, tmp_merges = 0;
        uint8_t tmp[SIZE][SIZE];
        copy_board(tmp, board);
        // initial move
        int init_ok = move(tmp, (enum move_type)m, &tmp_score, &tmp_merges);
        if (!init_ok) continue;

        for (int i = 0; i < num_iter; i++) {
            total_score += random_run(tmp);
        }

        scores[m] = total_score;
    }

    int max = scores[0], max_i = 0;
    for (int i = 1; i < 4; i++) {
        if (scores[i] > max) {
            max = scores[i];
            max_i = i;
        }
    }

    move(board, max_i, score, num_merges);
}

// Function to write the header for a CSV file
void write_csv_header(FILE *file)
{
    fprintf(file,
            "Game Number,Number of Moves,Score,Largest Tile,Sum of "
            "Tiles, Number of Merges, Losing Configuration,seconds\n");
}

// Function to write data to a CSV file
void write_csv_row(FILE *file, int game_number, int num_moves, int score,
                   int largest_tile, int sum_of_tiles, int num_merges,
                   const uint8_t losing_config[SIZE][SIZE], double time)
{
    fprintf(file,
            "%d,%d,%d,%d,%d,%d,\"%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%"
            "d\",%f\n",
            game_number, num_moves, score, largest_tile, sum_of_tiles, num_merges,
            losing_config[0][0], losing_config[0][1], losing_config[0][2],
            losing_config[0][3], losing_config[1][0], losing_config[1][1],
            losing_config[1][2], losing_config[1][3], losing_config[2][0],
            losing_config[2][1], losing_config[2][2], losing_config[2][3],
            losing_config[3][0], losing_config[3][1], losing_config[3][2],
            losing_config[3][3], time);
}
// monte_carlo_game(num_branch_to_explore, display, &num_moves, &score,
//                  &largest, &sum, &num_merges, &final_config
void monte_carlo_game(int num_branch_to_explore, bool display, int *num_moves,
                      int *final_score, int *largest, int *sum, int *num_merges,
                      uint8_t final_config[SIZE][SIZE])
{
    uint8_t board[SIZE][SIZE];
    uint32_t score = 0;
    uint32_t tmp_merges = 0;
    initBoard(board);
    if (display) drawBoard(board, 0, score);
    int i = 0;
    while (!(gameEnded(board))) {
        monte_carlo_iter(board, &score, &tmp_merges, num_branch_to_explore);
        if (display) {
            drawBoard(board, 0, score);
            #if !(defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__))
            usleep(1000 * 5);
            #endif
        }
        i++;
    }
    setBufferedInput(true);

    // make cursor visible, reset all modes
    printf("\033[?25h\033[m");

    printf("Game ended in %d moves. Score: %d. Largest tile: %d\n", i, score,
           max_tile(board));

    // save data to write to csv
    *num_moves = i;
    *final_score = score;
    *num_merges = tmp_merges;
    *largest = max_tile(board);
    *sum = sum_tile(board);
    copy_board(final_config, board);
}

void monte_carlo_simulation(int num_branch_to_explore, int num_games,
                            bool display)
{
#if METHOD == 0
    printf("Using max method\n");
#elif METHOD == 1
    // printf("Using merge method\n");
#elif METHOD == 2
    printf("Using sum method\n");
#else
    printf("Using weighted sum method\n");
#endif

    char fn[100] = {0};

    sprintf(fn, "data/monte_carlo_branch=%d_ngames=%d_method=%d.csv",
            num_branch_to_explore, num_games, METHOD);
    FILE *csv = fopen(fn, "w");
    write_csv_header(csv);
    for (int i = 0; i < num_games; i++) {
        int num_moves, score, largest, sum, num_merges;
        uint8_t final_config[SIZE][SIZE];
        clock_t start, end;
        start = clock();
        monte_carlo_game(num_branch_to_explore, display, &num_moves, &score,
                         &largest, &sum, &num_merges, final_config);
        end = clock();
        double diff = ((double)(end - start)) / CLOCKS_PER_SEC;
        write_csv_row(csv, i, num_moves, score, largest, sum, num_merges, final_config,
                      diff);
    }
    fclose(csv);
    printf("Wrote data to './%s'\n", fn);
}

void random_game()
{
    uint8_t board[SIZE][SIZE];
    uint32_t score = 0;
    uint32_t num_merges = 0;
    initBoard(board);
    setBufferedInput(false);
    while (!(gameEnded(board))) {
        bool ok = move(board, (uint32_t)rand() % 4, &score, &num_merges);
        if (ok) {
            addRandom(board);
        }
    }
    setBufferedInput(true);

    // make cursor visible, reset all modes
    // printf("\033[?25h\033[m");
    // printf("Random player got score %d, max tile %d\n", score,
        //    1 << max_tile(board));
    printf("%d, %d\n", score, max_tile(board));
}

void random_simulation(int num_games)
{
    for (int i = 0; i < num_games; i++) {
        random_game();
    }
}

int main(int argc, char *argv[])
{
    uint8_t board[SIZE][SIZE];
    uint8_t scheme = 0;
    uint32_t score = 0;
    uint32_t num_merges = 0;
    char c;
    bool success;
    srand(time(NULL));
    // register signal handler for when ctrl-c is pressed
    signal(SIGINT, signal_callback_handler);

    if (argc == 2 && strcmp(argv[1], "test") == 0) {
        return test();
    }
    if (argc >= 2 && strcmp(argv[1], "mc") == 0) {
        if (argc < 5) {
            printf(
                "Usage ./2048 mc <num rollouts> <num games> <show games "
                "true/false>\n");
            return 1;
        }
        int num_rollouts = atoi(argv[2]);
        int num_games = atoi(argv[3]);
        int display = strcmp(argv[4], "true") == 0;
        printf("Doing %d rollouts per move\n", num_rollouts);
        printf("Doing %d games\n", num_games);
        printf("Display: %s\n", display ? "on" : "off");
        monte_carlo_simulation(num_rollouts, num_games, display);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "random") == 0) {
        random_simulation(NUM_GAMES);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "blackwhite") == 0) {
        scheme = 1;
    }
    if (argc == 2 && strcmp(argv[1], "bluered") == 0) {
        scheme = 2;
    }

    // make cursor invisible, erase entire screen
    printf("\033[?25l\033[2J");

    initBoard(board);
    setBufferedInput(false);
    drawBoard(board, scheme, score);
    while (true) {
        c = getchar();
        if (c == -1) {
            puts("\nError! Cannot read keyboard input!");
            break;
        }
        switch (c) {
        case 97:   // 'a' key
        case 104:  // 'h' key
        case 68:   // left arrow
            success = moveLeft(board, &score, &num_merges);
            break;
        case 100:  // 'd' key
        case 108:  // 'l' key
        case 67:   // right arrow
            success = moveRight(board, &score, &num_merges);
            break;
        case 119:  // 'w' key
        case 107:  // 'k' key
        case 65:   // up arrow
            success = moveUp(board, &score, &num_merges);
            break;
        case 115:  // 's' key
        case 106:  // 'j' key
        case 66:   // down arrow
            success = moveDown(board, &score, &num_merges);
            break;
        default:
            success = false;
        }
        if (success) {
            drawBoard(board, scheme, score);
            #if !(defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__))
            usleep(150 * 1000);  // 150 ms
            #endif
            addRandom(board);
            drawBoard(board, scheme, score);
            if (gameEnded(board)) {
                printf("         GAME OVER          \n");
                break;
            }
        }
        if (c == 'q') {
            printf("        QUIT? (y/n)         \n");
            c = getchar();
            if (c == 'y') {
                break;
            }
            drawBoard(board, scheme, score);
        }
        if (c == 'r') {
            printf("       RESTART? (y/n)       \n");
            c = getchar();
            if (c == 'y') {
                initBoard(board);
                score = 0;
            }
            drawBoard(board, scheme, score);
        }
    }
    setBufferedInput(true);

    // make cursor visible, reset all modes
    printf("\033[?25h\033[m");

    return EXIT_SUCCESS;
}
