
const sections = {
    "Data": [
        "UsePreparedData", "RawDataVersion", "InclusionFileVersion", 
        "DataBalancingMethod", "BalancingGroups", "NumPerBalancedGroup"
    ], 
    "Grouping": [
        "AgeGroups", "SexSeparated"
    ], 
    "Feature Selection": [
        "UsePreviouslySelectedFeatures", "FeatureOrientations", 
        "FeatureSelectionMethod", "FSThresholdMethod", "MaxFeatureNum"
    ], 
    "Model Training": [
        "UsePredefinedSplit", "TestsetRatio", 
        "UsePretrainedModels", "IncludedOptimizationModels", "Seed"
    ], 
    "Age Correction": [
        "AgeCorrectionMethod", "AgeCorrectionGroups"
    ]
}; 

function createBlock(title, keys, data) {
    const block = document.createElement("div"); 
    block.className = "block"; 

    const h2 = document.createElement("h2"); 
    h2.textContent = title; 
    block.appendChild(h2); 

    keys.forEach(key => {
        const value = data[key]; 
        const div = document.createElement("div"); 
        const label = document.createElement("div"); 
        label.className = "label"; 
        label.style.fontWeight = "bold"; 
        label.textContent = key + ": "; 
        div.appendChild(label); 

        if (value === null || value === undefined) {
            const val = document.createElement("div"); 
            val.className = "null";
            val.textContent = "--"; 
            div.appendChild(val); 

        } else if (Array.isArray(value)) {
            const ul = document.createElement("ul"); 
            value.forEach(item => {
                const li = document.createElement("li"); 
                li.textContent = item; 
                ul.appendChild(li); 
            });
            div.appendChild(ul); 

        } else {
            const val = document.createElement("div"); 
            val.textContent = value; 
            div.appendChild(val); 
        }

        block.appendChild(div); 
    });

    return block; 
}

document.addEventListener("DOMContentLoaded", () => {
  const output = document.getElementById("output");
  for (const [sectionTitle, keys] of Object.entries(sections)) {
    const block = createBlock(sectionTitle, keys, config);
    output.appendChild(block);
  }
});

// document.addEventListener("DOMContentLoaded", () => { // Ensure the DOM is fully loaded before running the script
//     const fileInput = document.getElementById("fileInput"); 
//     const outputDiv = document.getElementById("output"); 

//     fileInput.addEventListener("change", (event) => {
//         const file = event.target.files[0]; 
//         if (!file) {
//             outputDiv.innerHTML = "<p>No file selected.</p>"; 
//             return; 
//         }

//         const reader = new FileReader(); 
//         reader.onload = function(e) {
//             try {
//                 const data = JSON.parse(e.target.result); // Parse the JSON data
//                 outputDiv.innerHTML = ""; // Clear previous output

//                 for (const [sectionTitle, keys] of Object.entries(sections)) {
//                     const block = createBlock(sectionTitle, keys, data); 
//                     outputDiv.appendChild(block); 
//                 }

//             } catch (error) {
//                 alert("Invalid JSON format."); 
//                 console.error(err); 
//             }
//         };

//         reader.readAsText(file); 
//     }); 
// });