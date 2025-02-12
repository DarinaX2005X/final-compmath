# Numerical Methods Calculator

## Overview
Our program is a user-friendly GUI built with Tkinter that allows users to select from a list of numerical methods, enter the required parameters, and execute the chosen algorithm. Once a method is selected, the corresponding input fields appear, and users can provide their data (such as functions, matrices, or initial guesses). When the Calculate button is pressed, the program processes the input, runs the selected numerical method, and displays the results in a text area along with any relevant graphs on a separate tab. The application includes methods such as the bisection method, Newton-Raphson, relaxation, power method, exponential fit, spline interpolation, Picard’s method, Simpson’s rule, and error analysis.

## Features
### Dynamic Method Selection:
Users can choose a numerical method from a dropdown list. The interface dynamically displays the relevant input fields for that method.

### Intuitive User Interface:
The GUI is designed using Tkinter, providing clear instructions, dynamic input fields, and separate areas for textual results and graphical outputs.

### Robust Error Handling:
Each algorithm includes error checking to catch issues like invalid inputs, division by zero, or non-convergence, and these errors are communicated to the user through both the text output area and popup messages.

### Accurate Numerical Methods:
The project implements a variety of numerical methods that produce correct results as per the project requirements.

### Visual Results Presentation:
Graphs and plots are generated using Matplotlib and embedded directly within the GUI to enhance understanding of the results.

## Installation
### Prerequisites
Make sure you have Python 3.x installed. Additionally, install the following Python libraries:

- numpy
- matplotlib
- scipy
- sympy

You can install these dependencies using pip:
```bash
pip install numpy matplotlib scipy sympy
```

## Usage
Run the program using:
```bash
python Final.py
```

1. Select a numerical method from the list.
2. Enter the required parameters in the input fields.
3. Click the "Calculate" button to execute the method.
4. View results in the output area or graphical tab.

## Code Structure
The code is organized around a single main class, MathApp, which encapsulates both the graphical user interface (GUI) and the numerical algorithms.

### Main Components
#### GUI Initialization
- Tkinter Setup: The `__init__` method of the MathApp class configures the main window, sets the title and dimensions, and initializes the main container.

#### Method Selection
- Method ComboBox: A combobox allows users to choose a numerical method from a predefined list. When a method is selected, the `update_inputs` method dynamically creates the corresponding input fields.

#### Input & Output Areas
- Text and Graphical Outputs: The application features two main output areas:
  - A text area for displaying results.
  - A separate tab for graphical outputs (using Matplotlib embedded via FigureCanvasTkAgg).

#### User Input & Execution
- Dynamic Input Fields: The `update_inputs` method is responsible for generating input fields based on the selected method.
- Execution Process: When the user clicks the Calculate button, the `execute_method` function:
  - Collects input values via the `get_input_values` method.
  - Clears any previous results.
  - Calls the appropriate numerical algorithm function (e.g., `handle_bisection`, `handle_newton_raphson`, etc.).

#### Error Handling
Each method implementation includes try-except blocks to catch and display errors both in the results text area and through popup messages.

### Numerical Algorithms
Each numerical method (such as Bisection, Newton-Raphson, Relaxation, Power, Exponential Fit, Spline Interpolation, Picard’s Method, and Simpson’s Rule) is implemented as a separate function within the MathApp class. These functions:

- Parse the user inputs.
- Execute the corresponding algorithm.
- Check for and handle errors (e.g., invalid input formats or non-convergence).
- Display detailed results and generate graphs when needed.

### Visualization
Graphs and plots are created using Matplotlib and are displayed on a dedicated tab. This integration is achieved through the FigureCanvasTkAgg widget.

## Detailed Method Descriptions
### Absolute/Relative Errors
- Computes the absolute error (the absolute difference between the true value and the approximate value) and the relative error (the ratio of the absolute error to the true value).

### Graphical Method
- Plots a user-specified function over a given range to visually identify features like roots or intersections.

### Bisection Method
- Finds a root of a function by repeatedly bisecting an interval where the function changes sign.

### Newton-Raphson Method
- Iteratively finds a root using the function’s derivative and a provided initial guess.

### Relaxation Method
- Solves a system of equations by iteratively updating an initial guess using a relaxation parameter (ω).
- This method includes checks for convergence and ensures that diagonal elements are sufficiently large to avoid division by zero.

### Power Method
- Finds the largest eigenvalue and corresponding eigenvector of a matrix by iteratively multiplying by an initial vector and normalizing.

### Exponential Fit
- Fits an exponential model to given data points using curve fitting (implemented with SciPy’s curve_fit).

### Spline Interpolation
- Uses either cubic or quadratic splines to interpolate data and estimate values at points between given data points.

### Picard’s Method
- Uses Picard’s successive approximations to iteratively solve an ordinary differential equation (ODE).

### Simpson’s Rule
- Approximates the definite integral of a function using Simpson’s rule.

## Usage Example
### Select a Method
Use the dropdown list at the top of the application to choose a numerical method (e.g., "Bisection Method").

### Enter Parameters
The interface will dynamically display the required input fields. For the bisection method, you might enter:
- Function: `x**3 - 6*x**2 + 11*x - 6`
- Left boundary (a): `0`
- Right boundary (b): `3`
- Tolerance: `1e-6`

### Execute the Calculation
Click the Calculate button. The program:
- Processes your inputs.
- Runs the selected algorithm.
- Displays the results (such as the approximate root and the number of iterations) in the Results tab.
- If applicable, a graph is displayed in the Graph/Plot tab.

### Error Handling
If there is any issue (e.g., invalid input or non-convergence), an error message is displayed both in the results area and as a popup message.
