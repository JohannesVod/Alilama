import matplotlib.pyplot as plt

def read_data(file_path):
    iterations = []
    eval_losses = []
    samples = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("Iteration:"):
                tokens = line.split(',')
                iteration = int(tokens[0].split(':')[1].strip())
                eval_loss = float(tokens[1].split(':')[1].strip())
                sample_tokens = tokens[3].split('=')[1].strip().split('x')
                samples_read = int(sample_tokens[0].split(" ")[0])
                iterations.append(iteration)
                eval_losses.append(eval_loss)
                samples.append(samples_read)
    return iterations, eval_losses, samples

# List of file paths
file_paths = ['models/alilama_batches_512_470K.pth_loss.txt', 
              'models/alilama_batches_32_470K.pth_loss.txt', ]  # Add more file paths as needed

# Plot the data from each file
plt.figure(figsize=(10, 6))
for file_path in file_paths:
    iterations, eval_losses, samples = read_data(file_path)
    plt.plot(samples, eval_losses, marker='o', linestyle='-', label=file_path)

plt.title('Evaluation Loss vs. Tokens Read')
plt.xlabel('Tokens Read')
plt.ylabel('Evaluation Loss')
plt.grid(True)
plt.legend()
plt.show()
