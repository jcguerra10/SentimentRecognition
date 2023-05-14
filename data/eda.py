import matplotlib.pyplot as plt
import pandas as pd
import textstat

def plot_label_histogram(df: pd.DataFrame) -> None:
    """
    Plots a histogram of the label column of a given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to plot the histogram of.

    Returns
    -------
    None

    """
    # Get the list of labels
    label_list = df['priority'].tolist()

    # Plot the histogram
    plt.hist(label_list, bins=7)
    plt.title('priority distribution')
    plt.show()

def plot_text_grade_distribution(df: pd.DataFrame) -> None:
    """
    Plots a histogram of the text grade level of a given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to plot the histogram of.

    Returns
    -------
    None

    """
    # Get the list of text grade levels
    grade_list = df['text'].apply(lambda x: textstat.text_standard(x)).tolist()

    # Plot the histogram
    plt.hist(grade_list, bins=7)
    plt.title('text grade distribution')
    plt.show()

def plot_text_length_distribution(df: pd.DataFrame) -> None:
    """
    Plots a histogram of the text length of a given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to plot the histogram of.

    Returns
    -------
    None

    """
    # Get the list of text lengths
    length_list = df['text'].apply(len).tolist()

    # Plot the histogram
    plt.hist(length_list, bins=7)
    plt.title('text length distribution')
    plt.show()

def plot_text_sentence_distribution(df: pd.DataFrame) -> None:
    """
    Plots a histogram of the text sentence count of a given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to plot the histogram of.

    Returns
    -------
    None

    """
    # Get the list of text sentence counts
    sentence_list = df['text'].apply(lambda x: textstat.sentence_count(x)).tolist()

    # Plot the histogram
    plt.hist(sentence_list, bins=7)
    plt.title('text sentence distribution')
    plt.show()

def plot_text_reading_time_distribution(df: pd.DataFrame) -> None:
    """
    Plots a histogram of the text reading time of a given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to plot the histogram of.

    Returns
    -------
    None

    """
    # Get the list of text reading times
    reading_time_list = df['text'].apply(lambda x: textstat.reading_time(x)).tolist()

    # Plot the histogram
    plt.hist(reading_time_list, bins=7)
    plt.title('text reading time distribution')
    plt.show()