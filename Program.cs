
using System;
using System.Globalization;
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

    public class TeamCapacityModel
    {
        [LoadColumn(0)]
        public DateTime Date;
        [LoadColumn(1)]
        public float Team_Capacity;
        [LoadColumn(2)]
        public float Dept_Capacity;
        [LoadColumn(3)]
        public float Approval_Rate;
        [LoadColumn(4)]
        public string Manager_Id;

        public static TeamCapacityModel FromCsv(string csvLine)
        {
            string[] values = csvLine.Split(',');
            TeamCapacityModel dailyValues = new TeamCapacityModel();
            dailyValues.Date = Convert.ToDateTime(values[0]);
            dailyValues.Team_Capacity = float.Parse(values[1], CultureInfo.InvariantCulture.NumberFormat);
            dailyValues.Dept_Capacity = float.Parse(values[2], CultureInfo.InvariantCulture.NumberFormat);
            dailyValues.Approval_Rate = float.Parse(values[3], CultureInfo.InvariantCulture.NumberFormat);
            dailyValues.Manager_Id = values[4];
            return dailyValues;
        }
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

            string leavePath = "C:/D Drive/Abhilash/Projects/Prediction/Prediction/OpenAidata.csv";
            string teamCapacityData = "C:/D Drive/Abhilash/Projects/Prediction/Prediction/TeamCapacityData.csv";

            MLContext mlContext = new MLContext();

            // 2. Load training data
            IDataView trainData = mlContext.Data.LoadFromTextFile<LeaveDataInput>(leavePath, separatorChar: ',');
            List<TeamCapacityModel> values = File.ReadAllLines(teamCapacityData)
                                       .Skip(1)
                                       .Select(v => TeamCapacityModel.FromCsv(v))
                                       .ToList();
            

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

            // data from user
            Console.WriteLine($"Enter leave date:");
            var userDateInput = Console.ReadLine();

            var convertedDate = DateTime.Parse(userDateInput, null, DateTimeStyles.None);
            var masterData = values.Find(x => x.Date == convertedDate);

            if (convertedDate < DateTime.Today || float.Parse(DateTime.Today.Subtract(convertedDate).TotalDays.ToString()) > 90)
            {
                //Console.WriteLine($"ML value: {result.LeaveStatus}\n");
                Console.WriteLine($"High Chances that your manager will REJECT");
                return;
            }

            if (masterData == null && convertedDate > DateTime.Today)
            {
                // we are assuming that master data in not present for that date and full team capacity is available
                Console.WriteLine($"Cannot predict approval probability for this date due to lack of past data.");
                return;
            }

            // 7. Predict on sample data and print results
            var input = new LeaveDataInput
            {
                ManagerId = masterData.Manager_Id,
                LeaveDateInterval = float.Parse(convertedDate.Subtract(DateTime.Today).TotalDays.ToString()),
                RoleCapacity = masterData.Dept_Capacity,
                TeamCapacity = masterData.Team_Capacity
            };

            var result = mlContext.Model.CreatePredictionEngine<LeaveDataInput, LeaveDataOutput>(model).Predict(input);
            if (result.LeaveStatus > 0)
            {
                Console.WriteLine($"ML value: {result.LeaveStatus}\n");
                Console.WriteLine($"High Chances that your manager will APPROVE");
            }
            else
            {
                Console.WriteLine($"ML value: {result.LeaveStatus}\n");
                Console.WriteLine($"High Chances that your manager will REJECT");
            }

        }
    }
}