from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, TextInput
from bokeh.layouts import column

# Define the ColumnDataSource
source = ColumnDataSource(data=dict(x=[1.0], y=[2.0]))  # Initialize with default values

# Define the scatter plot function
def create_scatter_plot():
    p = figure(width=500, height=250)
    p.scatter(x='x', y='y', source=source, size=10, color='blue')
    return p

# Callback function to update the plot
def update_plot(attr, old, new):
    try:
        x_pos = float(x_input.value)
        y_pos = float(y_input.value)
        source.data = {'x': [x_pos], 'y': [y_pos]}  # Replace the existing data
        plot.title.text = f"Scatter Plot (Position: {x_pos},{y_pos})"
    except ValueError:
        pass  # Invalid input, do nothing

# Create plot
plot = create_scatter_plot()

# Create input widgets
x_input = TextInput(value="1.0", title="X Position:")
y_input = TextInput(value="2.0", title="Y Position:")

# Attach the update function to the input widgets
x_input.on_change('value', update_plot)
y_input.on_change('value', update_plot)

# Arrange widgets and plot in a layout
layout = column(x_input, y_input, plot)

# Add the layout to the current document
curdoc().add_root(layout)
# bokeh serve --show interact.py