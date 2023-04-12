
using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;

namespace LeaveApprovalPrediction
{
    //class LeaveData
    //{
    //    [LoadColumn(0)]
    //    public float EmployeeId { get; set; }

    //    [LoadColumn(1)]
    //    public string JobTitle { get; set; }

    //    [LoadColumn(2)]
    //    public string Department { get; set; }

    //    [LoadColumn(3)]
    //    public float LengthOfService { get; set; }

    //    [LoadColumn(4)]
    //    public string Reason { get; set; }

    //    [LoadColumn(5)]
    //    public bool ManagerApproval { get; set; }
    //}

    //public class ModelInput
    //{
    //    [LoadColumn(2)]
    //    public float PassengerCount;
    //    [LoadColumn(3)]
    //    public float TripTime;
    //    [LoadColumn(4)]
    //    public float TripDistance;
    //    [LoadColumn(5)]
    //    public string PaymentType;
    //    [LoadColumn(6)]
    //    public float FareAmount;
    //}

    //public class ModelOutput
    //{
    //    [ColumnName("Score")]
    //    public float FareAmount;
    //}


    public class TrainerData
    {
        [LoadColumn(0)]
        public string ManagerId;
        [LoadColumn(1)]
        public float LeaveDateInterval;
        [LoadColumn(2)]
        public float TeamCapacity;
        [LoadColumn(3)]
        public float RoleCapacity;
        [LoadColumn(4)]
        public float LeaveStatus;
    }

    public class ModelOutput
    {
        [ColumnName("Status")]
        public float Status;
    }

    //class LeavePrediction
    //{
    //    [ColumnName("PredictedLabel")]
    //    public bool Prediction { get; set; }
    //}

    class Program
    {
        static void Main(string[] args)
        {
            //https://www.codemag.com/Article/1911042/ML.NET-Machine-Learning-for-.NET-Developers
            // Create a new MLContext
            MLContext mlContext = new MLContext();
            
            // Load the data
            IDataView dataView = mlContext.Data.LoadFromTextFile<TrainerData>("C:/D Drive/Abhilash/Projects/Prediction/Prediction/TrainingFile.csv", separatorChar: ',');

            //// Split the data into a training set and a test set
            //TrainTestData trainTestSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.3);
            //IDataView trainingData = trainTestSplit.TrainSet;
            //IDataView testData = trainTestSplit.TestSet;

            //// Define the pipeline
            //var pipeline = mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "JobTitleEncoded", inputColumnName: "JobTitle")
            //    .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "DepartmentEncoded", inputColumnName: "Department"))
            //    .Append(mlContext.Transforms.Concatenate("Features", "EmployeeId", "JobTitleEncoded", "DepartmentEncoded", "LengthOfService", "Reason"))
            //    .Append(mlContext.Transforms.NormalizeMinMax("Features"))
            //    .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression());

            //// Train the model
            //var model = pipeline.Fit(trainingData);

            //// Make predictions on the test data
            //var predictions = model.Transform(testData);
            //var metrics = mlContext.BinaryClassification.Evaluate(predictions);

            //Console.WriteLine($"Accuracy: {metrics.Accuracy}");
            //Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve}");

            //// Use the model to predict whether a new leave request will be approved by the manager or not
            //var predictionEngine = mlContext.Model.CreatePredictionEngine<LeaveData, LeavePrediction>(model);
            //var newRequest = new LeaveData { EmployeeId = 2, JobTitle = "Manager", Department = "Sales", LengthOfService = 5, Reason = "Vacation" };
            //var prediction = predictionEngine.Predict(newRequest);

            //Console.WriteLine($"Prediction: {prediction.Prediction}");



            // 3. Add data transformations
            var dataProcessPipeline = mlContext.Transforms.Categorical.OneHotEncoding(
                outputColumnName: "ManagerIdEncoded", "ManagerId")
                .Append(mlContext.Transforms.Concatenate(outputColumnName: "Features",
                "ManagerIdEncoded", "LeaveDateInterval", "TeamCapacity", "RoleCapacity"));


            //SdcaLogisticRegression
            // 4. Add algorithm
            var trainer = mlContext.Regression.Trainers.Sdca(labelColumnName: "LeaveStatus", featureColumnName: "Features");

            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // 5. Train model
            var model = trainingPipeline.Fit(dataView);

            // 6. Evaluate model on test data
            IDataView testData = mlContext.Data.LoadFromTextFile<TrainerData>("C:/D Drive/Abhilash/Projects/Prediction/Prediction/TrainingFile.csv");
            IDataView predictions = model.Transform(testData);
            var metrics = mlContext.Regression.Evaluate(predictions, "LeaveStatus");

            // 7. Predict on sample data and print results
            var input = new TrainerData
            {
                ManagerId = "M1",
                LeaveDateInterval = 10,
                TeamCapacity = 75,
                RoleCapacity = 75
            };

            var result = mlContext.Model.CreatePredictionEngine<TrainerData, ModelOutput>(model).Predict(input);

            Console.WriteLine($"Predicted Status: {result.Status}\n");
        }
    }
}