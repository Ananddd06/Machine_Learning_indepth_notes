import numpy as np

def compute_cost(X, y, theta):
    """Computes the Mean Squared Error (MSE)."""
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return cost

def gradient_descent_with_early_stopping(X_train, y_train, X_val, y_val, learning_rate=0.01, n_epochs=5000, patience=10):
    """
    Performs batch gradient descent with early stopping.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data for early stopping.
        learning_rate (float): The learning rate for gradient descent.
        n_epochs (int): The maximum number of epochs to run.
        patience (int): How many epochs to wait for improvement before stopping.

    Returns:
        tuple: The best theta found and the history of training/validation losses.
    """
    m_train = len(y_train)
    
    # Add intercept term to X
    X_train_b = np.c_[np.ones((m_train, 1)), X_train]
    X_val_b = np.c_[np.ones((len(y_val), 1)), X_val]

    # Randomly initialize theta
    theta = np.random.randn(X_train_b.shape[1], 1)
    
    # Early stopping parameters
    best_validation_loss = float('inf')
    best_theta = None
    patience_counter = 0
    
    train_loss_history = []
    val_loss_history = []

    print(f"Starting training for up to {n_epochs} epochs with patience={patience}...")

    for epoch in range(n_epochs):
        # --- Gradient Descent Step on Training Data ---
        gradients = (1 / m_train) * X_train_b.T.dot(X_train_b.dot(theta) - y_train)
        theta = theta - learning_rate * gradients
        
        # --- Evaluation on Training and Validation Data ---
        train_loss = compute_cost(X_train_b, y_train, theta)
        validation_loss = compute_cost(X_val_b, y_val, theta)
        
        train_loss_history.append(train_loss)
        val_loss_history.append(validation_loss)

        # --- Early Stopping Check ---
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_theta = theta.copy() # Save a copy of the best theta
            patience_counter = 0 # Reset patience
            # print(f"Epoch {epoch}: Validation loss improved to {validation_loss:.4f}")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEARLY STOPPING! No improvement in validation loss for {patience} epochs.")
            print(f"Stopped at epoch {epoch}. Best validation loss: {best_validation_loss:.4f}")
            break
            
    if best_theta is None: # Handle case where training finishes before any improvement
        best_theta = theta.copy()

    print("\nTraining finished.")
    return best_theta, train_loss_history, val_loss_history

# --- Example Usage ---
if __name__ == '__main__':
    # 1. Generate some sample data
    np.random.seed(42)
    X = 2 * np.random.rand(200, 1)
    y = 4 + 3 * X + np.random.randn(200, 1) # y = 4 + 3x + noise

    # 2. Split data into training (60%), validation (20%), and test (20%)
    train_size = int(0.6 * len(X))
    val_size = int(0.2 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    # 3. Run Gradient Descent with Early Stopping
    final_theta, train_loss, val_loss = gradient_descent_with_early_stopping(
        X_train, y_train, X_val, y_val, learning_rate=0.05, n_epochs=1000, patience=20
    )

    print(f"\nBest theta found:\n{final_theta}")

    # 4. Evaluate on test set with the best theta
    X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]
    test_loss = compute_cost(X_test_b, y_test, final_theta)
    print(f"\nFinal Test Loss (MSE) with best_theta: {test_loss:.4f}")

