
using System;
using System.Linq;
using System.Reflection;
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

    public class ModelInput
    {
        [LoadColumn(2)]
        public float PassengerCount;
        [LoadColumn(3)]
        public float TripTime;
        [LoadColumn(4)]
        public float TripDistance;
        [LoadColumn(5)]
        public string PaymentType;
        [LoadColumn(6)]
        public float FareAmount;
    }

    public class ModelOutput
    {
        [ColumnName("Score")]
        public float FareAmount;
    }


    public class LeaveDataInput
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
    public class LeaveDataOutput
    {
        [ColumnName("Score")]
        public float LeaveStatus;
    }


    //"C:/D Drive/Abhilash/Projects/Prediction/Prediction/TrainingFile.csv"
    //public class ModelOutput
    //{
    //    [ColumnName("Status")]
    //    public float Status;
    //}

    //class LeavePrediction
    //{
    //    [ColumnName("PredictedLabel")]
    //    public bool Prediction { get; set; }
    //}

    class Program
    {
        static void Main(string[] args)
        {
            
            string trainPath = "C:/D Drive/Abhilash/Projects/Prediction/Prediction/taxi-fare-train.csv";
            string leavePath = "C:/D Drive/Abhilash/Projects/Prediction/Prediction/OpenAidata.csv";

            if (true)
            {
                MLContext mlContext = new MLContext();

                // 2. Load training data
                IDataView trainData = mlContext.Data.LoadFromTextFile<LeaveDataInput>(leavePath, separatorChar: ',');

                // 3. Add data transformations
                var dataProcessPipeline = mlContext.Transforms.Categorical.OneHotEncoding(
                    outputColumnName: "ManagerIdEncoded", "ManagerId")
                    .Append(mlContext.Transforms.Concatenate(outputColumnName: "Features",
                    "ManagerIdEncoded", "LeaveDateInterval", "TeamCapacity", "RoleCapacity"));

                // 4. Add algorithm
                var trainer = mlContext.Regression.Trainers.Sdca(labelColumnName: "LeaveStatus", featureColumnName: "Features");

                var trainingPipeline = dataProcessPipeline.Append(trainer);

                // 5. Train model
                var model = trainingPipeline.Fit(trainData);

                // 6. Evaluate model on test data
                IDataView testData = mlContext.Data.LoadFromTextFile<LeaveDataInput>(leavePath);
                IDataView predictions = model.Transform(testData);
                var metrics = mlContext.Regression.Evaluate(predictions, "LeaveStatus");

                // 7. Predict on sample data and print results
                var input = new LeaveDataInput
                {
                    ManagerId = "M1",
                    LeaveDateInterval = 77,
                    RoleCapacity = 0.69f,
                    TeamCapacity = 0.68f
                };

                var result = mlContext.Model.CreatePredictionEngine<LeaveDataInput, LeaveDataOutput>(model).Predict(input);

                Console.WriteLine($"Predicted fare: {result.LeaveStatus}\n");
            }
            else
            {
                //https://www.codemag.com/Article/1911042/ML.NET-Machine-Learning-for-.NET-Developers
                // Create a new MLContext
                // 1. Initalize ML.NET environment
                MLContext mlContext = new MLContext();

                // 2. Load training data
                IDataView trainData = mlContext.Data.LoadFromTextFile<ModelInput>(trainPath, separatorChar: ',');

                // 3. Add data transformations
                var dataProcessPipeline = mlContext.Transforms.Categorical.OneHotEncoding(
                    outputColumnName: "PaymentTypeEncoded", "PaymentType")
                    .Append(mlContext.Transforms.Concatenate(outputColumnName: "Features",
                    "PaymentTypeEncoded", "PassengerCount", "TripTime", "TripDistance"));

                // 4. Add algorithm
                var trainer = mlContext.Regression.Trainers.Sdca(labelColumnName: "FareAmount", featureColumnName: "Features");

                var trainingPipeline = dataProcessPipeline.Append(trainer);

                // 5. Train model
                var model = trainingPipeline.Fit(trainData);

                // 6. Evaluate model on test data
                IDataView testData = mlContext.Data.LoadFromTextFile<ModelInput>(trainPath);
                IDataView predictions = model.Transform(testData);
                var metrics = mlContext.Regression.Evaluate(predictions, "FareAmount");

                // 7. Predict on sample data and print results
                var input = new ModelInput
                {
                    PassengerCount = 1,
                    TripTime = 1150,
                    TripDistance = 4,
                    PaymentType = "CRD"
                };

                var result = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model).Predict(input);

                Console.WriteLine($"Predicted fare: {result.FareAmount}\nModel Quality (RSquared): {metrics.RSquared}");
            }
            
        }
    }
}