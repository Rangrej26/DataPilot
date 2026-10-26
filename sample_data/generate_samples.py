"""
DataPilot AI - Sample Dataset Generator
Creates realistic academic and business datasets containing intentional real-world data flaws:
- Missing values
- Duplicate records
- Inconsistent text and casing
- Mixed date formats
- Outliers
- Target columns for Classification and Regression
- Multi-sheet Excel workbook
- 50,000-row benchmark dataset for performance testing
"""

import os
import random
import numpy as np
import pandas as pd

def generate_ecommerce_dataset(output_path: str):
    """Generates an E-Commerce Customer Churn dataset with messy real-world flaws."""
    np.random.seed(42)
    random.seed(42)
    n = 1500

    customer_ids = [f"CUST-{1000 + i}" for i in range(n)]
    ages = np.random.randint(18, 70, size=n).astype(float)
    # Inject missing ages
    ages[np.random.choice(n, size=50, replace=False)] = np.nan

    genders = np.random.choice(["Male", "Female", "Other", None], size=n, p=[0.48, 0.47, 0.03, 0.02])
    
    # Incomes with realistic distribution and extreme outliers
    incomes = np.random.normal(65000, 25000, size=n).round(-2)
    # Outliers
    incomes[12] = 950000.0   # Extreme high outlier
    incomes[88] = -12000.0   # Negative invalid income
    incomes[204] = 1200000.0 # Extreme high outlier
    incomes[np.random.choice(n, size=65, replace=False)] = np.nan  # Missing values

    # Spending score (1 - 100)
    spending_scores = np.random.randint(1, 101, size=n).astype(float)
    spending_scores[np.random.choice(n, size=40, replace=False)] = np.nan

    membership_years = np.random.choice([0.5, 1, 2, 3, 5, 7, 10], size=n)

    # Inconsistent categories with case issues
    categories_pool = [
        "Electronics", "electronics", "ELECTRONICS ", "Fashion", "fashion", 
        "Home & Kitchen", "home & kitchen", "Beauty", "Books", "Sports"
    ]
    preferred_category = np.random.choice(categories_pool, size=n)
    
    total_purchases = np.random.poisson(lam=18, size=n)

    # Inconsistent cities with spaces and casing
    cities_pool = [
        " New York ", "new york", "New York", "Los Angeles", "los angeles", 
        "Chicago", " Chicago ", "Houston", "houston", "Phoenix"
    ]
    cities = np.random.choice(cities_pool, size=n)

    # Mixed date formats
    dates = []
    base_dates = pd.date_range("2023-01-01", "2024-06-30", freq="D")
    for i in range(n):
        if i % 30 == 0:
            dates.append("Invalid Date")
        elif i % 25 == 0:
            dates.append(None)
        elif i % 4 == 0:
            d = random.choice(base_dates)
            dates.append(d.strftime("%d/%m/%Y"))
        elif i % 4 == 1:
            d = random.choice(base_dates)
            dates.append(d.strftime("%Y-%m-%d"))
        elif i % 4 == 2:
            d = random.choice(base_dates)
            dates.append(d.strftime("%b %d, %Y"))
        else:
            d = random.choice(base_dates)
            dates.append(d.strftime("%m-%d-%Y"))

    discount_used = np.random.choice(["Yes", "No", "yes", "NO"], size=n)
    
    # Constant column (data flaw to be detected by cleaning engine)
    constant_col = ["Active_Batch_2024"] * n

    # Churned target variable (dependent on spending score and membership)
    prob_churn = 1 / (1 + np.exp(-(-0.03 * np.nan_to_num(spending_scores, nan=50) + 0.2 * (membership_years < 2))))
    churned = (np.random.rand(n) < prob_churn).astype(int)

    df = pd.DataFrame({
        "Customer_ID": customer_ids,
        "Age": ages,
        "Gender": genders,
        "Annual_Income": incomes,
        "Spending_Score": spending_scores,
        "Membership_Years": membership_years,
        "Preferred_Category": preferred_category,
        "Total_Purchases": total_purchases,
        "City": cities,
        "Last_Purchase_Date": dates,
        "Discount_Used": discount_used,
        "Tracking_Code": constant_col,
        "Churned": churned
    })

    # Inject 42 duplicate rows
    duplicate_rows = df.iloc[:42].copy()
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    
    df.to_csv(output_path, index=False)
    print(f"Generated {output_path} with {len(df)} rows and {len(df.columns)} columns.")


def generate_employee_excel(output_path: str):
    """Generates a multi-sheet Excel file with Employee & Department data."""
    np.random.seed(101)
    n = 850

    departments = ["Engineering", "Sales", "Marketing", "Human Resources", "Finance", "Product"]
    dept_choices = np.random.choice(departments, size=n, p=[0.35, 0.25, 0.15, 0.08, 0.10, 0.07])
    
    emp_ids = [f"EMP-{2000 + i}" for i in range(n)]
    first_names = ["Alex", "Sam", "Jordan", "Taylor", "Morgan", "Casey", "Avery", "Riley", "Peyton", "Dakota"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
    names = [f"{random.choice(first_names)} {random.choice(last_names)}" for _ in range(n)]
    
    roles = {
        "Engineering": ["Software Engineer", "DevOps Engineer", "QA Engineer", "Data Engineer"],
        "Sales": ["Sales Executive", "Account Manager", "Sales Lead"],
        "Marketing": ["Content Specialist", "SEO Analyst", "Marketing Lead"],
        "Human Resources": ["HR Generalist", "Recruiter", "HR Director"],
        "Finance": ["Financial Analyst", "Accountant", "Finance Manager"],
        "Product": ["Product Manager", "UX Designer"]
    }
    job_roles = [random.choice(roles[dept]) for dept in dept_choices]

    monthly_salary = []
    for dept in dept_choices:
        base = 5000 if dept == "Engineering" else 4000
        salary = np.random.normal(base, 1200)
        monthly_salary.append(round(max(2000, salary), 2))
    monthly_salary = np.array(monthly_salary)
    # Inject missing salaries
    monthly_salary[np.random.choice(n, size=35, replace=False)] = np.nan
    # Outliers
    monthly_salary[5] = 45000.0

    years_at_co = np.random.randint(0, 15, size=n)
    perf_rating = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.05, 0.15, 0.50, 0.20, 0.10])
    overtime = np.random.choice(["Yes", "No"], size=n, p=[0.3, 0.7])
    education = np.random.choice(["Bachelor's", "Master's", "PhD", "High School"], size=n, p=[0.6, 0.25, 0.05, 0.1])
    
    # Attrition (Yes/No)
    prob_attrition = 0.15 + (overtime == "Yes") * 0.15 - (perf_rating >= 4) * 0.08
    attrition = np.where(np.random.rand(n) < prob_attrition, "Yes", "No")

    df_employees = pd.DataFrame({
        "Emp_ID": emp_ids,
        "Full_Name": names,
        "Department": dept_choices,
        "Job_Role": job_roles,
        "Monthly_Salary": monthly_salary,
        "Years_At_Company": years_at_co,
        "Performance_Rating": perf_rating,
        "OverTime": overtime,
        "Education_Level": education,
        "Attrition": attrition
    })

    # Add 15 duplicates
    df_employees = pd.concat([df_employees, df_employees.iloc[:15]], ignore_index=True)

    # Sheet 2: Departments
    df_dept = pd.DataFrame({
        "Department": departments,
        "Dept_Head": ["Elena Rostova", "Marcus Vance", "Sarah Chen", "David Kim", "Amira Patel", "Lucas Silva"],
        "Annual_Budget_M": [4.5, 3.2, 1.8, 0.9, 1.4, 2.1],
        "Location_Floor": ["Floor 4", "Floor 2", "Floor 3", "Floor 1", "Floor 2", "Floor 3"],
        "Cost_Center": ["CC-ENG-101", "CC-SLS-201", "CC-MKT-301", "CC-HR-401", "CC-FIN-501", "CC-PRD-601"]
    })

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_employees.to_excel(writer, sheet_name="Employees", index=False)
        df_dept.to_excel(writer, sheet_name="Department_Info", index=False)

    print(f"Generated multi-sheet {output_path} with sheets: Employees, Department_Info.")


def generate_patient_health_dataset(output_path: str):
    """Generates Medical Patient Records suitable for both Regression and Multi-class classification."""
    np.random.seed(2024)
    n = 1200

    patient_ids = [f"PAT-{3000 + i}" for i in range(n)]
    ages = np.random.randint(20, 80, size=n)
    
    bmi = np.random.normal(27.5, 5.0, size=n).round(1)
    bmi[bmi < 15] = 18.5
    bmi[10] = 78.4  # outlier
    bmi[45] = 95.0  # outlier

    bp = np.random.normal(122, 15, size=n).round(0)
    glucose = np.random.normal(105, 30, size=n).round(0)
    cholesterol = np.random.normal(195, 40, size=n).round(0)
    
    smoking = np.random.choice(["Never", "Former", "Current"], size=n, p=[0.55, 0.25, 0.20])
    activity_hours = np.random.gamma(shape=2, scale=1.5, size=n).round(1)
    family_history = np.random.choice([0, 1], size=n, p=[0.65, 0.35])

    # Continuous target: Health_Risk_Score (10 - 100) -> Regression Target
    risk_score = (
        0.3 * ages + 
        1.2 * bmi + 
        0.2 * bp + 
        0.15 * glucose + 
        (smoking == "Current") * 12 + 
        family_history * 8 - 
        activity_hours * 2.5
    )
    risk_score = np.clip(np.round(risk_score, 1), 10, 100)

    # Multi-class target: Diabetic_Status -> Classification Target
    # Normal (<100 glucose), Pre-diabetic (100-125), Diabetic (>125)
    diabetic_status = []
    for g in glucose:
        if g < 100:
            diabetic_status.append("Normal")
        elif g <= 125:
            diabetic_status.append("Pre-diabetic")
        else:
            diabetic_status.append("Diabetic")

    df = pd.DataFrame({
        "Patient_ID": patient_ids,
        "Age": ages,
        "BMI": bmi,
        "Blood_Pressure": bp,
        "Glucose_Level": glucose,
        "Cholesterol": cholesterol,
        "Smoking_Status": smoking,
        "Physical_Activity_Hours": activity_hours,
        "Family_History_Diabetes": family_history,
        "Health_Risk_Score": risk_score,
        "Diabetic_Status": diabetic_status
    })

    # Missing values
    df.loc[np.random.choice(n, size=30, replace=False), "Cholesterol"] = np.nan
    df.loc[np.random.choice(n, size=25, replace=False), "BMI"] = np.nan

    df.to_csv(output_path, index=False)
    print(f"Generated {output_path} with {len(df)} rows.")


def generate_benchmark_50k(output_path: str):
    """Generates a 50,000 row transaction dataset for testing high-speed DuckDB queries and caching."""
    np.random.seed(999)
    n = 50000

    tx_ids = [f"TX-{100000 + i}" for i in range(n)]
    dates = pd.date_range("2023-01-01", "2024-12-31", periods=n)
    stores = np.random.choice([f"STR-{100 + i}" for i in range(25)], size=n)
    regions = np.random.choice(["North America", "Europe", "Asia Pacific", "Latin America", "Middle East"], size=n)
    products = np.random.choice(["Laptops", "Smartphones", "Tablets", "Accessories", "Smartwatches", "Audio"], size=n)
    units = np.random.randint(1, 10, size=n)
    
    price_map = {
        "Laptops": 999.99,
        "Smartphones": 699.99,
        "Tablets": 449.99,
        "Accessories": 29.99,
        "Smartwatches": 249.99,
        "Audio": 149.99
    }
    unit_prices = [price_map[p] for p in products]
    revenue = np.round(units * np.array(unit_prices), 2)
    payments = np.random.choice(["Credit Card", "PayPal", "Apple Pay", "Debit Card", "Bank Transfer"], size=n)

    df = pd.DataFrame({
        "Transaction_ID": tx_ids,
        "Date": dates.strftime("%Y-%m-%d"),
        "Store_ID": stores,
        "Region": regions,
        "Product_Line": products,
        "Units_Sold": units,
        "Unit_Price": unit_prices,
        "Revenue": revenue,
        "Payment_Method": payments
    })

    df.to_csv(output_path, index=False)
    print(f"Generated {output_path} with {len(df)} rows.")


if __name__ == "__main__":
    os.makedirs("sample_data", exist_ok=True)
    generate_ecommerce_dataset("sample_data/ecommerce_churn.csv")
    generate_employee_excel("sample_data/employee_attrition.xlsx")
    generate_patient_health_dataset("sample_data/patient_health_risk.csv")
    generate_benchmark_50k("sample_data/performance_benchmark_50k.csv")
    print("All sample datasets successfully generated!")
