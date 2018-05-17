
def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [s+t for s in A for t in B]

assignments = []
rows = 'ABCDEFGHI'
cols = '123456789'

boxes = cross(rows, cols)
row_units = [cross(r, cols) for r in rows]
column_units = [cross(rows, c) for c in cols]
square_units = [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')]
# define diagonal units
diagonal_units = [[rows[i]+cols[i] for i in range(9)],[rows[i]+cols[-i-1] for i in range(9)]]
# add diagonal units to unitlist
unitlist = row_units + column_units + square_units + diagonal_units
units = dict((s, [u for u in unitlist if s in u]) for s in boxes)
peers = dict((s, set(sum(units[s],[]))-set([s])) for s in boxes)

def assign_value(values, box, value):
    """
    Please use this function to update your values dictionary!
    Assigns a value to a given box. If it updates the board record it.
    """

    # Don't waste memory appending actions that don't actually change any values
    if values[box] == value:
        return values

    values[box] = value
    if len(value) == 1:
        assignments.append(values.copy())
    return values

def naked_twins(values):
    """Eliminate values using the naked twins strategy.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the values dictionary with the naked twins eliminated from peers.
    """
    # ==============Find all instances of naked twins=================
    # Get all the boxes with 2 potential values
    potential_twins = [box for box in values.keys() if len(values[box]) == 2]

    # Compare all those potential_twins in their own peers
    # if their potential values are the same, append them into the array.
    naked_twins = [[box1,box2] for box1 in potential_twins \
                    for box2 in peers[box1] \
                    if set(values[box1])==set(values[box2]) ]

    for i in range(len(naked_twins)):
        box1 = naked_twins[i][0]
        box2 = naked_twins[i][1]
        # Intersection: peers[box1] ∩ peers[box2]
        peers1 = set(peers[box1]) # Marge the same item in different peers
        peers2 = set(peers[box2]) # Marge the same item in different peers
        peers_int = peers1 & set(peers[box2]) # Get all peers of naked_twins.

    # ======Eliminate the naked twins as possibilities for their peers=====
    # Delete the two digits in naked twins from all common peers.
        for peer_val in peers_int:
            if len(values[peer_val])>2: # More than 2 to make sure we can remove at least 1
                for rm_val in values[box1]:
                    values = assign_value(values, peer_val, values[peer_val].replace(rm_val,''))

    return values


def grid_values(grid):
    """
    Convert grid into a dict of {square: char} with '123456789' for empties.
    Args:
        grid(string) - A grid in string form.
    Returns:
        A grid in dictionary form
            Keys: The boxes, e.g., 'A1'
            Values: The value in each box, e.g., '8'. If the box has no value, then the value will be '123456789'.
    """
    chars = []
    all_digits = '123456789'
    for c in grid:
        if c == '.':
            chars.append(all_digits)
        elif c in all_digits:
            chars.append(c)
    assert len(chars) == 81, "Input grid must be a string of length 81 (9x9)"
    # boxes = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'B1', 'B2',....]
    # grid = ..3.2.6..9..3.5..1..18.64....81.29..7.......8..67.82....26.95..8..2.3..9..5.1.3..
    return dict(zip(boxes, chars))

def display(values):
    """
    Display the values as a 2-D grid.
    Args:
        values(dict): The sudoku in dictionary form
    """
    width = 1+max(len(values[s]) for s in boxes)
    line = '+'.join(['-'*(width*3)]*3)
    for r in rows:
        print(''.join(values[r+c].center(width)+('|' if c in '36' else '')
                      for c in cols))
        if r in 'CF': print(line)
    return

def eliminate(values):
    #syntactic sugar
    #https://stackoverflow.com/questions/6475314/python-for-in-loop-preceded-by-a-variable
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    # solved_values = ['A3', 'A5', 'A7', 'B1', 'B4', 'B6', 'B9', 'C3',....]
    for box in solved_values:
        digit = values[box]
        # peer 係本身utils 到set 左
        for peer in peers[box]:
            #values[peer] = values[peer].replace(digit,'')
            values = assign_value(values, peer, values[peer].replace(digit,''))
    return values

def only_choice(values):
    # unitlist =
    # [['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9'],['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9'], ...
    # ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G1', 'H1', 'I1'],['A2', 'B2', 'C2', 'D2', 'E2', 'F2', 'G2', 'H2', 'I2'], ...
    # ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3'],['A4', 'A5', 'A6', 'B4', 'B5', 'B6', 'C4', 'C5', 'C6']]
    for unit in unitlist:
        for digit in '123456789':
            # Sai: Get all boxes in current digit(1/2/3/4...) and iterated unit!
            dplaces = [box for box in unit if digit in values[box]]
            if len(dplaces) == 1:
                # values[dplaces[0]] = digit
                values = assign_value(values, dplaces[0], digit)
    return values

def reduce_puzzle(values):
    stalled = False
    while not stalled:
        # Check how many boxes have a determined value
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])

        # Your code here: Use the Eliminate Strategy
        values = eliminate(values)

        # Your code here: Use the Only Choice Strategy
        values = only_choice(values)

        # Naked twins technique
        values = naked_twins(values)

        # Check how many boxes have a determined value, to compare
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        # If no new values were added, stop the loop. (Sai: important!)
        stalled = solved_values_before == solved_values_after
        # Sanity check, return False if there is a box with zero available values: (Sai: important!)
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values

def search(values):
    # First, reduce the puzzle using the previous function
    values = reduce_puzzle(values)

    # Sai: this line is to make sure at least solved once
    if values is False:
        return False ## Failed earlier

    # boxes = 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'B1', 'B2', ....';
    # all: Return True if all elements of the iterable are true
    if all(len(values[s]) == 1 for s in boxes):
        return values ## Solved!

    # Choose one of the unfilled squares with the fewest possibilities
    n,s = min((len(values[s]), s) for s in boxes if len(values[s]) > 1)

    # Now use recursion to solve each one of the resulting sudokus, and if one returns a value (not False), return that answer!
    for value in values[s]:
        new_sudoku = assign_value(values, s, value)
        #如果用左reduce_puzzle 都搞唔到，就要行尼個recursive search 去估哂所有posibilities
        attempt = search(new_sudoku)
        if attempt:
            return attempt

def solve(grid):
    """
    Find the solution to a Sudoku grid.
    Args:
        grid(string): a string representing a sudoku grid.
            Example: '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    Returns:
        The dictionary representation of the final sudoku grid. False if no solution exists.
    """
    return search( reduce_puzzle(grid_values(grid)));

if __name__ == '__main__':
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    display(solve(diag_sudoku_grid))

    try:
        from visualize import visualize_assignments
        visualize_assignments(assignments)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
