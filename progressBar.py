def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    #print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    print("\r" + str(percent) + "% Completed.", end="")
    # Print New Line on Complete
    if iteration == total:
        print("\n")
