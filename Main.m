%% Project Title: Brain Tumor Detection

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
                
                % Data Augmentation
                % Randomly apply transformations
                if rand() > 0.5
                    I = imrotate(I, randi([-15, 15]));  % Random rotation
                end
                if rand() > 0.5
                    I = flip(I, 1);  % Random vertical flip
                end
                if rand() > 0.5
                    I = flip(I, 2);  % Random horizontal flip
                end
                if rand() > 0.5
                    % Randomly adjust brightness
                    I = imadjust(I, [], [], rand() * 0.2 + 0.9); % Adjust brightness
                end
                if rand() > 0.5
                    % Add Gaussian noise
                    I = imnoise(I, 'gaussian', 0, 0.01); % Add Gaussian noise
                end
                
                [I3, RGB] = createMask(I);
                seg_img = RGB;
                img = rgb2gray(seg_img);
                
                % Feature Extraction
                % Calculate GLCM and extract features
                glcms = graycomatrix(img);
                stats = graycoprops(glcms, 'Contrast Correlation Energy Homogeneity');
                Contrast = stats.Contrast;
                Energy = stats.Energy;
                Homogeneity = stats.Homogeneity;

                % Additional statistical features
                Mean = mean2(seg_img);
                Variance = var(double(seg_img(:)));
                Standard_Deviation = std2(seg_img);
                Skewness = skewness(double(seg_img(:)));
                Kurtosis = kurtosis(double(seg_img(:)));
                
                % GLCM features
                Dissimilarity = sum(sum(glcms)) - sum(diag(glcms)); % Dissimilarity
                
                % Calculate correlation from GLCM
                [rows, cols] = size(glcms);
                mu_x = sum((1:rows) .* sum(glcms, 2)); % Mean of x
                mu_y = sum((1:cols) .* sum(glcms, 1)); % Mean of y
                sigma_x = sqrt(sum(((1:rows) - mu_x).^2 .* sum(glcms, 2))); % Standard deviation of x
                sigma_y = sqrt(sum(((1:cols) - mu_y).^2 .* sum(glcms, 1))); % Standard deviation of y
                
                Correlation = (sum(sum((1:rows)' * (1:cols) .* glcms)) - mu_x * mu_y) / (sigma_x * sigma_y); % Correlation
                ASM = sum(sum(glcms.^2)); % Angular Second Moment
                Entropy = -sum(glcms(:) .* log(glcms(:) + eps)); % Entropy
                Coarseness = 1 / (1 + mean2(glcms)); % Coarseness

                % Combine features into a single row
                ff = [Mean, Variance, Standard_Deviation, Skewness, Kurtosis, Contrast, Energy, ASM, Entropy, Homogeneity, Dissimilarity, Correlation, Coarseness];
                Train_Feat = [Train_Feat; ff];  % Append to the feature matrix
                
                % Assign label based on the folder name
                Train_Label = [Train_Label; i];  % Use the index as the label
            end
        end

        % Check if training data is available
        if isempty(Train_Feat) || isempty(Train_Label)
            disp('Error: No training data available. Please ensure that the training images are correctly labeled and processed.');
            continue;  % Go back to the menu if no training data is available
        end

        disp('Training Complete');

        %% Hyperparameter
        % Define hyperparameters for the model
        numTrees = 100;  % Number of trees for Random Forest
        maxNumSplits = 2^maxDepth - 1;   % Maximum number of splits based on desired depth
        minLeafSize = 5; % Minimum leaf size

        % Train the Random Forest model
        model = TreeBagger(numTrees, Train_Feat, Train_Label, 'Method', 'classification', 'MaxNumSplits', maxNumSplits, 'MinLeafSize', minLeafSize);

        disp('Model training completed.');

        
        % Calculate performance metrics on training data
        predictions_train = predict(model, Train_Feat);

        % Convert predictions to numeric if Train_Label is numeric
        predictions_train_numeric = str2double(predictions_train); % Convert predictions to numeric

        % Calculate confusion matrix
        confusionMatrix_train = confusionmat(Train_Label, predictions_train_numeric);
        disp('Training Confusion Matrix:');
        disp(confusionMatrix_train);

        % Calculate accuracy
        accuracy_train = sum(diag(confusionMatrix_train)) / sum(confusionMatrix_train(:));
        fprintf('Training Accuracy: %.2f%%\n', accuracy_train * 100);

        % Calculate precision, recall, and F1 score for each class
        precision_train = zeros(1, length(classLabels));
        recall_train = zeros(1, length(classLabels));
        F1_score_train = zeros(1, length(classLabels));

        for i = 1:length(classLabels)
            TP = confusionMatrix_train(i, i); % True Positives
            FP = sum(confusionMatrix_train(:, i)) - TP; % False Positives
            FN = sum(confusionMatrix_train(i, :)) - TP; % False Negatives

            precision_train(i) = TP / (TP + FP + eps); % Precision
            recall_train(i) = TP / (TP + FN + eps); % Recall
            F1_score_train(i) = 2 * (precision_train(i) * recall_train(i)) / (precision_train(i) + recall_train(i) + eps); % F1 Score
        end

        % Display metrics for each class
        for i = 1:length(classLabels)
            fprintf('Training Class: %s\n', classLabels{i});
            fprintf('Precision: %.2f%%\n', precision_train(i) * 100);
            fprintf('Recall: %.2f%%\n', recall_train(i) * 100);
            fprintf('F1 Score: %.2f%%\n', F1_score_train(i) * 100);
        end

    elseif choice == 2
        %% Image Read for Testing
        Test_Feat = [];  % Initialize feature matrix for testing
        Test_Label = []; % Initialize label vector for testing
        basePathTest = 'C:/Users/paulj/OneDrive/Documents/GitHub/Brain-Tumor/Testing/';  

        % Loop through each test image
        testFiles = dir(fullfile(basePathTest, '*.jpg'));  % Get all jpg files in the testing folder
        disp(['Found ', num2str(length(testFiles)), ' images in Testing folder.']);  % Debugging statement

        for k = 1:length(testFiles)
            filePath = fullfile(basePathTest, testFiles(k).name);  % Full path to the image
            I = imread(filePath);
            I = imresize(I, [512, 512]);  % Resize to 512x512
            
            % Process the image similarly as in training
            [I3, RGB] = createMask(I );
            seg_img = RGB;
            img = rgb2gray(seg_img);
            
            % Feature Extraction for testing
            glcms = graycomatrix(img);
            stats = graycoprops(glcms, 'Contrast Correlation Energy Homogeneity');
            Contrast = stats.Contrast;
            Energy = stats.Energy;
            Homogeneity = stats.Homogeneity;

            % Additional statistical features
            Mean = mean2(seg_img);
            Variance = var(double(seg_img(:)));
            Standard_Deviation = std2(seg_img);
            Skewness = skewness(double(seg_img(:)));
            Kurtosis = kurtosis(double(seg_img(:)));
            
            % GLCM features
            Dissimilarity = sum(sum(glcms)) - sum(diag(glcms)); % Dissimilarity
            
            % Calculate correlation from GLCM
            [rows, cols] = size(glcms);
            mu_x = sum((1:rows) .* sum(glcms, 2)); % Mean of x
            mu_y = sum((1:cols) .* sum(glcms, 1)); % Mean of y
            sigma_x = sqrt(sum(((1:rows) - mu_x).^2 .* sum(glcms, 2))); % Standard deviation of x
            sigma_y = sqrt(sum(((1:cols) - mu_y).^2 .* sum(glcms, 1))); % Standard deviation of y
            
            Correlation = (sum(sum((1:rows)' * (1:cols) .* glcms)) - mu_x * mu_y) / (sigma_x * sigma_y); % Correlation
            ASM = sum(sum(glcms.^2)); % Angular Second Moment
            Entropy = -sum(glcms(:) .* log(glcms(:) + eps)); % Entropy
            Coarseness = 1 / (1 + mean2(glcms)); % Coarseness

            % Combine features into a single row for testing
            ff = [Mean, Variance, Standard_Deviation, Skewness, Kurtosis, Contrast, Energy, ASM, Entropy, Homogeneity, Dissimilarity, Correlation, Coarseness];
            Test_Feat = [Test_Feat; ff];  % Append to the feature matrix for testing
            Test_Label = [Test_Label; k];  % Assign a temporary label for testing
        end

        % Make predictions using the trained model
        if ~isempty(Test_Feat)
            predictions = predict(model, Test_Feat);
            disp('Testing Complete. Predictions made for test images.');

            % Calculate performance metrics for testing
            confusionMatrix_test = confusionmat(Test_Label, predictions);
            disp('Testing Confusion Matrix:');
            disp(confusionMatrix_test);

            % Calculate accuracy for testing
            accuracy_test = sum(diag(confusionMatrix_test)) / sum(confusionMatrix_test(:));
            fprintf('Testing Accuracy: %.2f%%\n', accuracy_test * 100);

            % Calculate precision, recall, and F1 score for each class in testing
            precision_test = zeros(1, length(classLabels));
            recall_test = zeros(1, length(classLabels));
            F1_score_test = zeros(1, length(classLabels));

            for i = 1:length(classLabels)
                TP = confusionMatrix_test(i, i); % True Positives
                FP = sum(confusionMatrix_test(:, i)) - TP; % False Positives
                FN = sum(confusionMatrix_test(i, :)) - TP; % False Negatives

                precision_test(i) = TP / (TP + FP + eps); % Precision
                recall_test(i) = TP / (TP + FN + eps); % Recall
                F1_score_test(i) = 2 * (precision_test(i) * recall_test(i)) / (precision_test(i) + recall_test(i) + eps); % F1 Score
            end

            % Display metrics for each class in testing
            for i = 1:length(classLabels)
                fprintf('Testing Class: %s\n', classLabels{i});
                fprintf('Precision: %.2f%%\n', precision_test(i) * 100);
                fprintf('Recall: %.2f%%\n', recall_test(i) * 100);
                fprintf('F1 Score: %.2f%%\n', F1_score_test(i) * 100);
            end

        else
            disp('Error: No test data available. Please ensure that the test images are correctly processed.');
        end

    elseif choice == 3
        disp('Exiting the program.');
        break;  % Exit the loop and close the program
    end
end