
CREATE TABLE Aspect
(
	AspectKey            integer NOT NULL,
	Name                 varchar(100) NOT NULL,
	Description          varchar(500) NULL
)
go

ALTER TABLE Aspect
ADD PRIMARY KEY (AspectKey)
go

CREATE TABLE BodyPart
(
	BodyPartKey          integer NOT NULL,
	Name                 varchar(100) NOT NULL,
	Description          varchar(500) NULL,
	StartSkeletonIndex   float NULL,
	EndSkeletonIndex     float NULL,
	StartSkeletonIndexDEPRECATED float NULL,
	EndSkeletonIndexDEPRECATED float NULL
)
go

ALTER TABLE BodyPart
ADD PRIMARY KEY (BodyPartKey)
go

CREATE TABLE Category
(
	CategoryKey          integer NOT NULL,
	Name                 varchar(100) NOT NULL,
	Description          varchar(500) NULL
)
go

ALTER TABLE Category
ADD PRIMARY KEY (CategoryKey)
go

CREATE TABLE ComputerVisionAlgorithm
(
	CVAlgorithmKey       integer NOT NULL,
	Name                 varchar(100) NOT NULL,
	Description          varchar(500) NULL,
	FrameByFrame         char(1) NOT NULL,
	Author               varchar(20) NULL,
	AcademicPaper        varchar(20) NULL,
	Code                 varchar(100) NOT NULL
)
go

ALTER TABLE ComputerVisionAlgorithm
ADD PRIMARY KEY (CVAlgorithmKey)
go

CREATE TABLE Direction
(
	DirectionKey         integer NOT NULL,
	Name                 varchar(20) NULL,
	Description          varchar(500) NULL
)
go

ALTER TABLE Direction
ADD PRIMARY KEY (DirectionKey)
go

CREATE TABLE Experimenter
(
	ExperimenterKey      integer NOT NULL,
	Name                 varchar(100) NOT NULL,
	Description          varchar(500) NULL,
	LabKey               integer NOT NULL,
	Update_User_Id       varchar(50) NOT NULL,
	Update_Timestamp     datetime NOT NULL
)
go

ALTER TABLE Experimenter
ADD PRIMARY KEY (ExperimenterKey)
go

CREATE TABLE Lab
(
	LabKey               integer NOT NULL,
	Name                 varchar(100) NOT NULL,
	Description          varchar(500) NULL,
	Address              varchar(20) NULL,
	Update_User_Id       varchar(50) NOT NULL,
	Update_Timestamp     datetime NOT NULL
)
go

ALTER TABLE Lab
ADD PRIMARY KEY (LabKey)
go

CREATE TABLE Plate
(
	PlateKey             integer NOT NULL,
	SampleType           char(18) NULL,
	StartDateTime        datetime NULL,
	Copyright            varchar(20) NULL,
	VulvaOrientation     varchar(20) NULL,
	Annotation           varchar(20) NULL,
	Chemicals            varchar(20) NULL,
	Food                 varchar(20) NULL,
	Illumination         varchar(20) NULL,
	Temperature          integer NULL,
	Tracker              varchar(20) NULL,
	AgarSide             varchar(20) NULL,
	GasConcentration     varbinary NULL,
	ExperimenterKey      integer NOT NULL,
	WormListKey          integer NOT NULL,
	Update_User_Id       varchar(50) NOT NULL,
	Update_Timestamp     datetime NOT NULL
)
go

ALTER TABLE Plate
ADD PRIMARY KEY (PlateKey)
go

CREATE TABLE PlateFeature
(
	PlateFeatureKey      integer NOT NULL,
	Name                 varchar(100) NOT NULL,
	Description          varchar(500) NULL,
	Title                varchar(20) NULL,
	ShortTitle           varchar(20) NULL
)
go

ALTER TABLE PlateFeature
ADD PRIMARY KEY (PlateFeatureKey)
go

CREATE TABLE PlateRawVideo
(
	PlateRawVideoKey     integer NOT NULL,
	VideoFile            varbinary NULL,
	PlateKey             integer NOT NULL,
	VideoMetadataKey     char(18) NOT NULL,
	Update_User_Id       varchar(50) NOT NULL,
	Update_Timestamp     datetime NOT NULL,
	Update_Timestamp__845 datetime NOT NULL
)
go

ALTER TABLE PlateRawVideo
ADD PRIMARY KEY (PlateRawVideoKey)
go

CREATE TABLE Sign
(
	SignKey              integer NOT NULL,
	Name                 varchar(100) NOT NULL,
	Description          varchar(500) NULL
)
go

ALTER TABLE Sign
ADD PRIMARY KEY (SignKey)
go

CREATE TABLE Strain
(
	StrainKey            integer NOT NULL,
	Strain_Name          varchar(100) NOT NULL,
	Gene                 varchar(20) NULL,
	Genotype             varbinary NULL,
	Allele               varchar(20) NULL,
	Chromosome           varchar(20) NULL,
	Simulated            char(1) NOT NULL,
	Update_User_Id       varchar(50) NOT NULL,
	Update_Timestamp     datetime NOT NULL
)
go

ALTER TABLE Strain
ADD PRIMARY KEY (StrainKey)
go

CREATE TABLE Type
(
	TypeKey              integer NOT NULL,
	Name                 varchar(100) NOT NULL,
	Description          varchar(500) NULL
)
go

ALTER TABLE Type
ADD PRIMARY KEY (TypeKey)
go

CREATE TABLE User
(
	UserID               integer NOT NULL,
	Name                 varchar(20) NULL,
	AccessLevel          float NULL,
	LabKey               integer NOT NULL
)
go

ALTER TABLE User
ADD PRIMARY KEY (UserID)
go

CREATE TABLE VideoAttributes
(
	VideoMetadataKey     char(18) NOT NULL,
	FPS                  integer NULL,
	NumFrames            float NULL,
	Width                integer NULL,
	Height               integer NULL,
	MicronsPerPixel      integer NULL,
	Update_User_Id       varchar(50) NOT NULL,
	Update_Timestamp     datetime NOT NULL
)
go

ALTER TABLE VideoAttributes
ADD PRIMARY KEY (VideoMetadataKey)
go

CREATE TABLE WormFeature
(
	WormFeatureKey       integer NOT NULL,
	Index                float NULL,
	Title                varchar(20) NULL,
	ShortTitle           varchar(20) NULL,
	Description          varchar(500) NULL,
	bin_width            integer NULL,
	is_signed            char(1) NOT NULL,
	is_time_series       char(1) NOT NULL,
	is_zero_bin          int NOT NULL,
	units                varchar(20) NULL,
	signed_field         varchar(20) NULL,
	remove_partial_events char(1) NOT NULL,
	make_zero_if_empty   char(1) NOT NULL,
	Name                 varchar(100) NOT NULL,
	TypeKey              integer NOT NULL,
	CategoryKey          integer NOT NULL,
	DirectionKey         integer NOT NULL,
	AspectKey            integer NOT NULL,
	BodyPartKey          integer NOT NULL
)
go

ALTER TABLE WormFeature
ADD PRIMARY KEY (WormFeatureKey)
go

CREATE TABLE Worm
(
	StrainKey            integer NOT NULL,
	Sex                  varchar(20) NULL,
	WormKey              integer NOT NULL,
	ThawedDate           datetime NULL,
	GenerationsSinceThawing float NULL,
	Habituation          varchar(20) NULL,
	Update_User_Id       varchar(50) NOT NULL,
	Update_Timestamp     datetime NOT NULL
)
go

ALTER TABLE Worm
ADD PRIMARY KEY (WormKey)
go

CREATE TABLE WormList
(
	WormListKey          integer NOT NULL,
	WormList_Identifier  int NOT NULL,
	WormKey              integer NOT NULL
)
go

ALTER TABLE WormList
ADD PRIMARY KEY (WormListKey)
go

CREATE TABLE PlateWireframeVideo
(
	PlateWireframeVideoKey integer NOT NULL,
	WireframeVideo       varbinary NULL,
	PlateRawVideoKey     integer NOT NULL,
	CVAlgorithmKey       integer NOT NULL,
	DroppedFrameInfo     varbinary NULL
)
go

ALTER TABLE PlateWireframeVideo
ADD PRIMARY KEY (PlateWireframeVideoKey)
go

CREATE TABLE WormInteraction
(
	WormInteractionKey   char(18) NOT NULL,
	FrameByFrameWormParticipation varbinary NULL,
	PlateWireframeVideoKey integer NOT NULL,
	WormListKey          integer NOT NULL,
	Area                 varbinary NULL,
	InteractionType      varchar(20) NULL,
	StartFrame           float NULL,
	EndFrame             float NULL
)
go

ALTER TABLE WormInteraction
ADD PRIMARY KEY (WormInteractionKey)
go

CREATE TABLE FeaturesPerPlateWireframe
(
	FeaturesPerPlateWireframe integer NOT NULL,
	Value                varbinary NULL,
	PlateFeatureKey      integer NOT NULL,
	PlateWireframeVideoKey integer NOT NULL
)
go

ALTER TABLE FeaturesPerPlateWireframe
ADD PRIMARY KEY (FeaturesPerPlateWireframe)
go

CREATE TABLE HistogramsPerPlateWireframe
(
	HistogramsPerPlateWireframeKey integer NOT NULL,
	Bins                 varbinary NULL,
	Counts               varbinary NULL,
	PlateWireframeVideoKey integer NOT NULL
)
go

ALTER TABLE HistogramsPerPlateWireframe
ADD PRIMARY KEY (HistogramsPerPlateWireframeKey)
go

CREATE TABLE WormWireframeVideo
(
	WormWireframeKey     integer NOT NULL,
	WireframeVideo       varbinary NULL,
	PlateWireframeVideoKey integer NOT NULL,
	DroppedFrameInfo     varbinary NULL
)
go

ALTER TABLE WormWireframeVideo
ADD PRIMARY KEY (WormWireframeKey)
go

CREATE TABLE HistogramsPerWormWireframe
(
	HistogramsPerWormWireframeKey integer NOT NULL,
	Bins                 varbinary NULL,
	Counts               varbinary NULL,
	SignKey              integer NOT NULL,
	EventDirectionKey    integer NOT NULL,
	WormFeatureKey       integer NOT NULL,
	WormWireframeKey     integer NOT NULL
)
go

ALTER TABLE HistogramsPerWormWireframe
ADD PRIMARY KEY (HistogramsPerWormWireframeKey)
go

CREATE TABLE FeaturesPerWormWireframe
(
	FeaturesPerWormWireframeKey integer NOT NULL,
	WormFeatureKey       integer NOT NULL,
	Value                varbinary NULL,
	WormWireframeKey     integer NOT NULL
)
go

ALTER TABLE FeaturesPerWormWireframe
ADD PRIMARY KEY (FeaturesPerWormWireframeKey)
go

CREATE TABLE WormMeasurement
(
	WormMeasurementsKey  integer NOT NULL,
	Name                 varchar(100) NOT NULL,
	Description          varchar(500) NULL
)
go

ALTER TABLE WormMeasurement
ADD PRIMARY KEY (WormMeasurementsKey)
go

CREATE TABLE MeasurementsPerWormWireframe
(
	MeasurementsPerWormWireframe integer NOT NULL,
	WormMeasurementsKey  integer NOT NULL,
	Value                varbinary NULL,
	WormWireframeKey     integer NOT NULL
)
go

ALTER TABLE MeasurementsPerWormWireframe
ADD PRIMARY KEY (MeasurementsPerWormWireframe)
go

ALTER TABLE Experimenter
ADD FOREIGN KEY R_48 (LabKey) REFERENCES Lab (LabKey)
go

ALTER TABLE Plate
ADD FOREIGN KEY R_11 (ExperimenterKey) REFERENCES Experimenter (ExperimenterKey)
go

ALTER TABLE Plate
ADD FOREIGN KEY R_46 (WormListKey) REFERENCES WormList (WormListKey)
go

ALTER TABLE PlateRawVideo
ADD FOREIGN KEY R_13 (PlateKey) REFERENCES Plate (PlateKey)
go

ALTER TABLE PlateRawVideo
ADD FOREIGN KEY R_28 (VideoMetadataKey) REFERENCES VideoAttributes (VideoMetadataKey)
go

ALTER TABLE User
ADD FOREIGN KEY R_49 (LabKey) REFERENCES Lab (LabKey)
go

ALTER TABLE WormFeature
ADD FOREIGN KEY R_5 (TypeKey) REFERENCES Type (TypeKey)
go

ALTER TABLE WormFeature
ADD FOREIGN KEY R_6 (CategoryKey) REFERENCES Category (CategoryKey)
go

ALTER TABLE WormFeature
ADD FOREIGN KEY R_7 (DirectionKey) REFERENCES Direction (DirectionKey)
go

ALTER TABLE WormFeature
ADD FOREIGN KEY R_8 (AspectKey) REFERENCES Aspect (AspectKey)
go

ALTER TABLE WormFeature
ADD FOREIGN KEY R_9 (BodyPartKey) REFERENCES BodyPart (BodyPartKey)
go

ALTER TABLE Worm
ADD FOREIGN KEY R_1 (StrainKey) REFERENCES Strain (StrainKey)
go

ALTER TABLE WormList
ADD FOREIGN KEY R_45 (WormKey) REFERENCES Worm (WormKey)
go

ALTER TABLE PlateWireframeVideo
ADD FOREIGN KEY R_14 (PlateRawVideoKey) REFERENCES PlateRawVideo (PlateRawVideoKey)
go

ALTER TABLE PlateWireframeVideo
ADD FOREIGN KEY R_15 (CVAlgorithmKey) REFERENCES ComputerVisionAlgorithm (CVAlgorithmKey)
go

ALTER TABLE WormInteraction
ADD FOREIGN KEY R_39 (PlateWireframeVideoKey) REFERENCES PlateWireframeVideo (PlateWireframeVideoKey)
go

ALTER TABLE WormInteraction
ADD FOREIGN KEY R_47 (WormListKey) REFERENCES WormList (WormListKey)
go

ALTER TABLE FeaturesPerPlateWireframe
ADD FOREIGN KEY R_35 (PlateFeatureKey) REFERENCES PlateFeature (PlateFeatureKey)
go

ALTER TABLE FeaturesPerPlateWireframe
ADD FOREIGN KEY R_36 (PlateWireframeVideoKey) REFERENCES PlateWireframeVideo (PlateWireframeVideoKey)
go

ALTER TABLE HistogramsPerPlateWireframe
ADD FOREIGN KEY R_41 (PlateWireframeVideoKey) REFERENCES PlateWireframeVideo (PlateWireframeVideoKey)
go

ALTER TABLE WormWireframeVideo
ADD FOREIGN KEY R_40 (PlateWireframeVideoKey) REFERENCES PlateWireframeVideo (PlateWireframeVideoKey)
go

ALTER TABLE HistogramsPerWormWireframe
ADD FOREIGN KEY R_16 (SignKey) REFERENCES Sign (SignKey)
go

ALTER TABLE HistogramsPerWormWireframe
ADD FOREIGN KEY R_17 (EventDirectionKey) REFERENCES Direction (DirectionKey)
go

ALTER TABLE HistogramsPerWormWireframe
ADD FOREIGN KEY R_27 (WormFeatureKey) REFERENCES WormFeature (WormFeatureKey)
go

ALTER TABLE HistogramsPerWormWireframe
ADD FOREIGN KEY R_38 (WormWireframeKey) REFERENCES WormWireframeVideo (WormWireframeKey)
go

ALTER TABLE FeaturesPerWormWireframe
ADD FOREIGN KEY R_20 (WormFeatureKey) REFERENCES WormFeature (WormFeatureKey)
go

ALTER TABLE FeaturesPerWormWireframe
ADD FOREIGN KEY R_34 (WormWireframeKey) REFERENCES WormWireframeVideo (WormWireframeKey)
go

ALTER TABLE MeasurementsPerWormWireframe
ADD FOREIGN KEY R_23 (WormMeasurementsKey) REFERENCES WormMeasurement (WormMeasurementsKey)
go

ALTER TABLE MeasurementsPerWormWireframe
ADD FOREIGN KEY R_37 (WormWireframeKey) REFERENCES WormWireframeVideo (WormWireframeKey)
go
