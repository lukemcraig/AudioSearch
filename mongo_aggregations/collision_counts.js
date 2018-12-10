db.getCollection("fingerprints").aggregate(

	// Pipeline
	[
		// Stage 1
		{
			$group: {
			_id :"$hash",
			count: { $sum: 1 },
			songs:{$addToSet:"$songID"},
			
			}
		},

		// Stage 2
		{
			$addFields: {
			    n_songs:{$size:"$songs"}
			}
		},

		// Stage 3
		{
			$sort: {
			n_songs:-1
			}
		},

	]
);
