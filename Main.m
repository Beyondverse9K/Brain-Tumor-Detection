%% Project Title: Brain Tumor Detection with CNN

while true
    choice = menu('Disease Detection', '....... Training........', '....... Testing......', '........ Close........');
    
    if choice == 1
        %% Image Read for Training
        Train_Feat = [];  % Initialize feature matrix
        Train_Label = []; % Initialize label vector
        basePath = 'C:/Users/paulj/OneDrive/Documents/GitHub/Brain-Tumor/Training/';  

        % Define the class labels and their corresponding folder names
        classLabels = {'glioma_tumor', 'meningioma_tumor', 'pituitary_tumor', 'no_tumor'};

        % Loop through each class folder
        for i = 1:length(classLabels)
            className = classLabels{i};
            classFolder = fullfile(basePath, className);  % Full path to the class folder
            imageFiles = dir(fullfile(classFolder, '*.jpg'));  % Get all jpg files in the folder

            disp(['Found ', num2str(length(imageFiles)), ' images in ', className, ' folder.']);  % Debugging statement

            % Loop through each image file in the class folder
            for k = 1:length(imageFiles)
                filePath = fullfile(classFolder, imageFiles(k).name);  % Full path to the image
                I = imread(filePath);
                I = imresize(I, [512, 512]);  % Resize to 512x512
                
                % Store the image and label
                Train_Feat(:,:,:,k) = I;  % Store image in 4D array
                Train_Label(k) = i;  % Use the index as the label
            end
        end

        % Check if training data is available
        if isempty(Train_Feat) || isempty(Train_Label)
            disp('Error: No training data available. Please ensure that the training images are correctly labeled and processed.');
            continue;  % Go back to the menu if no training data is available
        end

        disp('Training Complete');

        %% Define Data Augmentation
        augmenter = imageDataAugmenter( ...
            'RandRotation', [-15, 15], ...  % Random rotation between -15 and 15 degrees
            'RandXReflection', true, ...     % Random horizontal flip
            'RandYReflection', true, ...     % Random vertical flip
            'RandXTranslation', [-10, 10], ... % Random horizontal translation
            'RandYTranslation', [-10, 10]);   % Random vertical translation

        % Create an augmented image datastore
        augmentedTrainData = augmentedImageDatastore([512 512], Train_Feat, categorical(Train_Label), 'DataAugmentation', augmenter);

        %% Define CNN Architecture
        layers = [
            imageInputLayer([512 512 3])  % Input layer for 512x512 RGB images
            
            convolution2dLayer(5, 32, 'Padding', 'same')  % Convolutional layer
            batchNormalizationLayer  % Batch normalization
            reluLayer  % ReLU activation
            
            maxPooling2dLayer(2, 'Stride', 2)  % Max pooling layer
            
            convolution2dLayer(5, 64, 'Padding', 'same')  % Second convolutional layer
            batchNormalizationLayer
            reluLayer
            
            maxPooling2dLayer(2, 'Stride', 2)  % Second max pooling layer
            
            convolution2dLayer(5, 128, 'Padding', 'same')  % Third convolutional layer
            batchNormalizationLayer
            reluLayer
            
            maxPooling2dLayer(2, 'Stride', 2)  % Third max pooling layer
            
            fullyConnectedLayer(length(classLabels))  % Fully connected layer
            softmaxLayer  % Softmax layer
            classificationLayer  % Classification layer
        ];

        %% Training Options
        options = trainingOptions('adam', ...
            'MaxEpochs', 20, ...
            'Shuffle', 'every-epoch', ...
            'Verbose', false, ...
            'Plots', 'training-progress');

        %% Train the CNN
        net = trainNetwork(augmentedTrainData, layers, options);

        disp('CNN Model training completed.');

    elseif choice == 2
        %% Image Read for Testing
        [filename, pathname] = uigetfile({'*.*'; '*. bmp'; '*.jpg'; '*.gif'}, 'Pick a Tumor Image File');
        if isequal(filename, 0) || isequal(pathname, 0)
            disp('User  canceled the operation.');
            continue;  % Go back to the menu if no file is selected
        end
        
        I = imread(fullfile(pathname, filename));
        I = imresize(I, [512, 512]);  % Resize to match CNN input size
        figure, imshow(I); title('Query Tumor Image');
        
        %% Classify the Image
        predicted_class = classify(net, I);  % Use the trained CNN for prediction
        
        % Display the predicted class label
        disp(['Predicted Class: ', char(predicted_class)]);  

        % For performance metrics, you can implement similar logic as before
        actual_class = input('Enter the actual class index (1 for glioma, 2 for meningioma, 3 for pituitary, 4 for no tumor): ');

        % Calculate confusion matrix and performance metrics if needed
        confusionMatrix_test = confusionmat(actual_class, predicted_class);  % Create confusion matrix
        disp('Testing Confusion Matrix:');
        disp(confusionMatrix_test);

        % Calculate performance metrics
        TP = confusionMatrix_test(1, 1); % True Positives
        FP = confusionMatrix_test(1, 2) + confusionMatrix_test(1, 3) + confusionMatrix_test(1, 4); % False Positives
        FN = confusionMatrix_test(2, 1) + confusionMatrix_test(3, 1) + confusionMatrix_test(4, 1); % False Negatives

        % Accuracy
        accuracy = sum(diag(confusionMatrix_test)) / sum(confusionMatrix_test(:));
        
        % Precision
        precision = TP / (TP + FP + eps);  % Add epsilon to avoid division by zero
        
        % Recall
        recall = TP / (TP + FN + eps);
        
        % F1 Score
        if (precision + recall) > 0
            F1_score = 2 * (precision * recall) / (precision + recall);
        else
            F1_score = 0; % Avoid division by zero
        end

        % Display performance metrics
        fprintf('Accuracy: %.2f%%\n', accuracy * 100);
        fprintf('Precision: %.2f\n', precision);
        fprintf('Recall: %.2f\n', recall);
        fprintf('F1 Score: %.2f\n', F1_score);
    end

    if choice == 3
        disp('Exiting the program.');
        break;  % Exit the loop and close the program
    end
end