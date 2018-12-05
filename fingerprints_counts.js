db.getCollection("fingerprints").aggregate(

	// Pipeline
	[
		// Stage 1
		{
			$group: {
			_id :"$songID",
			count: { $sum: 1 }
			}
		},

		// Stage 2
		{
			$lookup: // Equality Match
			{
			    from: "songs",
			    localField: "_id",
			    foreignField: "_id",
			    as: "song"
			}
		},

		// Stage 3
		{
			$project: {
			    // specifications
			    count:1,
			    length:"$song.length"
			}
		},

	]
);
