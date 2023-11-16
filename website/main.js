function upload() {
	let fileInput = document.getElementById("csvFile");
	let file = fileInput.files[0];
	let reader = new FileReader();
	reader.readAsText(file);
	reader.onload = function() {
		let csv = reader.result;
		let table = "<table>";
		let rows = csv.split("\n");
		rows.forEach(function(row) {
			table += "<tr>";
			let cells = row.split(",");
			cells.forEach(function(cell) {
				table += "<td>" + cell + "</td>";
			});
			table += "</tr>";
		});
		table += "</table>";
		document.getElementById("table").innerHTML = table;
	};
}