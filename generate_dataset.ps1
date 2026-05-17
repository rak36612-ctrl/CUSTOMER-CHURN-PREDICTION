<#
.SYNOPSIS
Generates a random dataset for the Customer Churn Prediction Dashboard.
#>

$numRows = Read-Host "How many rows of data would you like to generate?"
$count = 0
if ([int]::TryParse($numRows, [ref]$count)) {
    if ($count -lt 50) {
        Write-Warning "The dashboard requires at least 50 rows to train the model. Setting rows to 50."
        $count = 50
    }
} else {
    Write-Warning "Invalid input. Generating 100 rows by default."
    $count = 100
}

Write-Host "Generating $count rows of customer data..." -ForegroundColor Cyan

$csv = "customerID,tenure,monthlyCharges,totalCharges,contract,internetService,churn`n"

for ($i = 1; $i -le $count; $i++) {
    $tenure = Get-Random -Minimum 1 -Maximum 73
    
    # Intentionally biasing churn to lower tenures for the ML model to recognize a pattern
    if ($tenure -lt 12) {
        $churn = Get-Random -InputObject "Yes", "Yes", "No"
    } else {
        $churn = Get-Random -InputObject "No", "No", "No", "Yes"
    }

    $monthly = (Get-Random -Minimum 2000 -Maximum 11500) / 100
    $total = [math]::Round($tenure * $monthly, 2)

    if ($tenure -gt 48) {
        $contract = "Two year"
    } elseif ($tenure -gt 12) {
        $contract = "One year"
    } else {
        $contract = "Month-to-month"
    }

    $internet = Get-Random -InputObject "Fiber optic", "DSL", "No"
    
    $csv += "CUST_$i,$tenure,$monthly,$total,$contract,$internet,$churn`n"
}

$filename = "generated_dataset.csv"
$csv | Out-File -FilePath $filename -Encoding utf8
Write-Host "Successfully generated '$filename' with $count rows!" -ForegroundColor Green
Write-Host "You can now drag and drop this file into your dashboard." -ForegroundColor Magenta
