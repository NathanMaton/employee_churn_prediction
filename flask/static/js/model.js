/* This function is called when the submit button is pressed/
 *
 * See the "on" attribute of the submit button */
function process(){
  console.log('process func called');

  // we have really long feature names =(
  feature = {
    'satisfaction_level': $('#satisfaction_level').val(),
    'last_evaluation': $('#last_evaluation').val(),
    'number_project':  $('#number_project').val(),
    'average_montly_hours': $('#average_montly_hours').val(),
    'time_spend_company': $('#time_spend_company').val()
     'Work_accident': 1,
     'promotion_last_5years': 0,
     'IT': 0,
     'RandD': 0,
     'accounting':0,
     'hr':0,
     'management':0,
     'marketing':0,
     'product_mng':0,
     'sales':0,
     'support':0,
     'technical':1,
     'high':0,
     'low':0,
     'medium':1
      }

  // Call our API route /predict_api via the post method
  // Our method returns a dictionary.
  // If successful, pass the dictionary to the function "metis_success"
  // If there is an error, pass the dictionary to the function "metis_error"
  // Note: functions can have any name you want; to demonstrate this we put
  //       metis_ at the beginning of each function.
  console.log(feature);
  // $.post({
  //   url: '/results',
  //   contentType: 'application/json',
  //   data: JSON.stringify(feature),
  //   success: result => metis_success(result),
  //   error: result => metis_error(result)
  // })
}

function ourRound(val, decimalPlaces=1){
  // Javascript rounds to integers by default, so this is a hack
  // to round to a certain number of decimalPlaces
  const factor = Math.pow(10, decimalPlaces)
  return Math.round(factor*val)/factor
}

/* Here "result" is the "dictionary" (javascript object)
 * that our get_api_response function returned when we called
 * the /predict_api function
 *
 * Here we select the "results" div and overwrite it
 */
function metis_success(result){
  alert("inside success");
  $('#results').html(`The most likely class is ${result.most_likely_class_name}
                      with probability ${ourRound(100*result.most_likely_class_prob)}%`);

  const all_results = result.all_probs.map( (data) => `${data.name}: ${ourRound(100*data.prob)}`)
  $('#list_results').html(all_results.join('%<br>') + '%');

  // only included in predictor_javascript_slider_graph.html
  // otherwise does nothing.
  modifyDivs(result.all_probs);
}

function metis_error(result){
  console.log(result);
  alert("Something's wrong.");
}
