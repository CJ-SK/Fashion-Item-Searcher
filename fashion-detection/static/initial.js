$(document).ready( function() {
  $("#loader").hide();
  $("#refresher").hide();
    } );

function readURL(input) {
  if (input.files && input.files[0]) {
    var reader = new FileReader();
    reader.onload = function (e) {
      $('#preview-img')
        .attr('src', e.target.result);
    };
    reader.readAsDataURL(input.files[0]);
  }
}

function upload() {
  // $("#input_pic").submit();
  $(".inner-div").hide();
  $("#loader").show();
  $("#refresher").show();
}
