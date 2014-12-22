
CREATE TYPE [Key]
	FROM INTEGER NULL
go

CREATE TYPE [Domain_183]
	FROM INT NOT NULL
go

CREATE TYPE [Name]
	FROM VARCHAR(100) NOT NULL
go

CREATE TYPE [Description]
	FROM VARCHAR(500) NULL
go

CREATE TYPE [Timestamp]
	FROM DATETIME NOT NULL
go

CREATE TYPE [Indicator]
	FROM CHAR(1) NOT NULL
go

CREATE TYPE [Amount]
	FROM FLOAT NULL
go

CREATE TYPE [Code]
	FROM VARCHAR(100) NOT NULL
go

CREATE TYPE [Update_Timestamp]
	FROM DATETIME NOT NULL
go

CREATE TYPE [Percent]
	FROM FLOAT NULL
go

CREATE TYPE [Identifier]
	FROM INT NOT NULL
go

CREATE TYPE [Count]
	FROM FLOAT NULL
go

CREATE TYPE [Update_User_Id]
	FROM VARCHAR(50) NOT NULL
go

CREATE TABLE [Aspect]
( 
	[AspectKey]          [Key]  NOT NULL ,
	[Name]               [Name] ,
	[Description]        [Description] 
)
go

ALTER TABLE [Aspect]
	ADD CONSTRAINT [XPKAspect] PRIMARY KEY  CLUSTERED ([AspectKey] ASC)
go

CREATE TABLE [BodyPart]
( 
	[BodyPartKey]        [Key]  NOT NULL ,
	[Name]               [Name] ,
	[Description]        [Description] ,
	[StartSkeletonIndex] [Count] ,
	[EndSkeletonIndex]   [Count] ,
	[StartSkeletonIndexDEPRECATED] [Count] ,
	[EndSkeletonIndexDEPRECATED] [Count] 
)
go

ALTER TABLE [BodyPart]
	ADD CONSTRAINT [XPKBodyPart] PRIMARY KEY  CLUSTERED ([BodyPartKey] ASC)
go

CREATE TABLE [Category]
( 
	[CategoryKey]        [Key]  NOT NULL ,
	[Name]               [Name] ,
	[Description]        [Description] 
)
go

ALTER TABLE [Category]
	ADD CONSTRAINT [XPKCategory] PRIMARY KEY  CLUSTERED ([CategoryKey] ASC)
go

CREATE TABLE [ComputerVisionAlgorithm]
( 
	[CVAlgorithmKey]     [Key]  NOT NULL ,
	[Name]               [Name] ,
	[Description]        [Description] ,
	[FrameByFrame]       [Indicator] ,
	[Author]             varchar(20)  NULL ,
	[AcademicPaper]      varchar(20)  NULL ,
	[Code]               [Code] 
)
go

ALTER TABLE [ComputerVisionAlgorithm]
	ADD CONSTRAINT [XPKComputerVisionAlgorithm] PRIMARY KEY  CLUSTERED ([CVAlgorithmKey] ASC)
go

CREATE TABLE [Direction]
( 
	[DirectionKey]       [Key]  NOT NULL ,
	[Name]               varchar(20)  NULL ,
	[Description]        [Description] 
)
go

ALTER TABLE [Direction]
	ADD CONSTRAINT [XPKDirection] PRIMARY KEY  CLUSTERED ([DirectionKey] ASC)
go

CREATE TABLE [Experimenter]
( 
	[ExperimenterKey]    [Key]  NOT NULL ,
	[Name]               [Name] ,
	[Description]        [Description] ,
	[LabKey]             [Key]  NOT NULL ,
	[Update_User_Id]     [Update_User_Id] ,
	[Update_Timestamp]   [Update_Timestamp] 
)
go

ALTER TABLE [Experimenter]
	ADD CONSTRAINT [XPKExperimenter] PRIMARY KEY  CLUSTERED ([ExperimenterKey] ASC)
go

CREATE TABLE [Lab]
( 
	[LabKey]             [Key]  NOT NULL ,
	[Name]               [Name] ,
	[Description]        [Description] ,
	[Address]            varchar(20)  NULL ,
	[Update_User_Id]     [Update_User_Id] ,
	[Update_Timestamp]   [Update_Timestamp] 
)
go

ALTER TABLE [Lab]
	ADD CONSTRAINT [XPKLab] PRIMARY KEY  CLUSTERED ([LabKey] ASC)
go

CREATE TABLE [Plate]
( 
	[PlateKey]           [Key]  NOT NULL ,
	[SampleType]         char(18)  NULL ,
	[StartDateTime]      datetime  NULL ,
	[Copyright]          varchar(20)  NULL ,
	[VulvaOrientation]   varchar(20)  NULL ,
	[Annotation]         varchar(20)  NULL ,
	[Chemicals]          varchar(20)  NULL ,
	[Food]               varchar(20)  NULL ,
	[Illumination]       varchar(20)  NULL ,
	[Temperature]        integer  NULL ,
	[Tracker]            varchar(20)  NULL ,
	[AgarSide]           varchar(20)  NULL ,
	[GasConcentration]   varbinary  NULL ,
	[ExperimenterKey]    [Key]  NOT NULL ,
	[WormListKey]        [Key]  NOT NULL ,
	[Update_User_Id]     [Update_User_Id] ,
	[Update_Timestamp]   [Update_Timestamp] 
)
go

ALTER TABLE [Plate]
	ADD CONSTRAINT [XPKPlate] PRIMARY KEY  CLUSTERED ([PlateKey] ASC)
go

CREATE TABLE [PlateFeature]
( 
	[PlateFeatureKey]    [Key]  NOT NULL ,
	[Name]               [Name] ,
	[Description]        [Description] ,
	[Title]              varchar(20)  NULL ,
	[ShortTitle]         varchar(20)  NULL 
)
go

ALTER TABLE [PlateFeature]
	ADD CONSTRAINT [XPKPlateFeature] PRIMARY KEY  CLUSTERED ([PlateFeatureKey] ASC)
go

CREATE TABLE [PlateRawVideo]
( 
	[PlateRawVideoKey]   [Key]  NOT NULL ,
	[VideoFile]          varbinary  NULL ,
	[PlateKey]           [Key]  NOT NULL ,
	[VideoMetadataKey]   char(18)  NOT NULL ,
	[Update_User_Id]     [Update_User_Id] ,
	[Update_Timestamp]   [Update_Timestamp] ,
	[Update_Timestamp__845] [Update_Timestamp] 
)
go

ALTER TABLE [PlateRawVideo]
	ADD CONSTRAINT [XPKPlateRawVideo] PRIMARY KEY  CLUSTERED ([PlateRawVideoKey] ASC)
go

CREATE TABLE [Sign]
( 
	[SignKey]            [Key]  NOT NULL ,
	[Name]               [Name] ,
	[Description]        [Description] 
)
go

ALTER TABLE [Sign]
	ADD CONSTRAINT [XPKSign] PRIMARY KEY  CLUSTERED ([SignKey] ASC)
go

CREATE TABLE [Strain]
( 
	[StrainKey]          [Key]  NOT NULL ,
	[Strain_Name]        [Name] ,
	[Gene]               varchar(20)  NULL ,
	[Genotype]           varbinary  NULL ,
	[Allele]             varchar(20)  NULL ,
	[Chromosome]         varchar(20)  NULL ,
	[Simulated]          [Indicator] ,
	[Update_User_Id]     [Update_User_Id] ,
	[Update_Timestamp]   [Update_Timestamp] 
)
go

ALTER TABLE [Strain]
	ADD CONSTRAINT [XPKStrain] PRIMARY KEY  CLUSTERED ([StrainKey] ASC)
go

CREATE TABLE [Type]
( 
	[TypeKey]            [Key]  NOT NULL ,
	[Name]               [Name] ,
	[Description]        [Description] 
)
go

ALTER TABLE [Type]
	ADD CONSTRAINT [XPKType] PRIMARY KEY  CLUSTERED ([TypeKey] ASC)
go

CREATE TABLE [User]
( 
	[UserID]             [Key]  NOT NULL ,
	[Name]               varchar(20)  NULL ,
	[AccessLevel]        [Count] ,
	[LabKey]             [Key]  NOT NULL 
)
go

ALTER TABLE [User]
	ADD CONSTRAINT [XPKUser] PRIMARY KEY  CLUSTERED ([UserID] ASC)
go

CREATE TABLE [VideoAttributes]
( 
	[VideoMetadataKey]   char(18)  NOT NULL ,
	[FPS]                integer  NULL ,
	[NumFrames]          [Count] ,
	[Width]              integer  NULL ,
	[Height]             integer  NULL ,
	[MicronsPerPixel]    integer  NULL ,
	[Update_User_Id]     [Update_User_Id] ,
	[Update_Timestamp]   [Update_Timestamp] 
)
go

ALTER TABLE [VideoAttributes]
	ADD CONSTRAINT [XPKVideoAttributes] PRIMARY KEY  CLUSTERED ([VideoMetadataKey] ASC)
go

CREATE TABLE [WormFeature]
( 
	[WormFeatureKey]     [Key]  NOT NULL ,
	[Index]              [Count] ,
	[Title]              varchar(20)  NULL ,
	[ShortTitle]         varchar(20)  NULL ,
	[Description]        [Description] ,
	[bin_width]          integer  NULL ,
	[is_signed]          [Indicator] ,
	[is_time_series]     [Indicator] ,
	[is_zero_bin]        [Identifier] ,
	[units]              varchar(20)  NULL ,
	[signed_field]       varchar(20)  NULL ,
	[remove_partial_events] [Indicator] ,
	[make_zero_if_empty] [Indicator] ,
	[Name]               [Name] ,
	[TypeKey]            [Key]  NOT NULL ,
	[CategoryKey]        [Key]  NOT NULL ,
	[DirectionKey]       [Key]  NOT NULL ,
	[AspectKey]          [Key]  NOT NULL ,
	[BodyPartKey]        [Key]  NOT NULL 
)
go

ALTER TABLE [WormFeature]
	ADD CONSTRAINT [XPKWormFeature] PRIMARY KEY  CLUSTERED ([WormFeatureKey] ASC)
go

CREATE TABLE [Worm]
( 
	[StrainKey]          [Key]  NOT NULL ,
	[Sex]                varchar(20)  NULL ,
	[WormKey]            [Key]  NOT NULL ,
	[ThawedDate]         datetime  NULL ,
	[GenerationsSinceThawing] [Count] ,
	[Habituation]        varchar(20)  NULL ,
	[Update_User_Id]     [Update_User_Id] ,
	[Update_Timestamp]   [Update_Timestamp] 
)
go

ALTER TABLE [Worm]
	ADD CONSTRAINT [XPKWorm] PRIMARY KEY  CLUSTERED ([WormKey] ASC)
go

CREATE TABLE [WormList]
( 
	[WormListKey]        [Key]  NOT NULL ,
	[WormList_Identifier] [Identifier] ,
	[WormKey]            [Key]  NOT NULL 
)
go

ALTER TABLE [WormList]
	ADD CONSTRAINT [XPKWormList] PRIMARY KEY  CLUSTERED ([WormListKey] ASC)
go

CREATE TABLE [PlateWireframeVideo]
( 
	[PlateWireframeVideoKey] [Key]  NOT NULL ,
	[WireframeVideo]     varbinary  NULL ,
	[PlateRawVideoKey]   [Key]  NOT NULL ,
	[CVAlgorithmKey]     [Key]  NOT NULL ,
	[DroppedFrameInfo]   varbinary  NULL 
)
go

ALTER TABLE [PlateWireframeVideo]
	ADD CONSTRAINT [XPKPlateWireframeVideo] PRIMARY KEY  CLUSTERED ([PlateWireframeVideoKey] ASC)
go

CREATE TABLE [WormInteraction]
( 
	[WormInteractionKey] char(18)  NOT NULL ,
	[FrameByFrameWormParticipation] varbinary  NULL ,
	[PlateWireframeVideoKey] [Key]  NOT NULL ,
	[WormListKey]        [Key]  NOT NULL ,
	[Area]               varbinary  NULL ,
	[InteractionType]    varchar(20)  NULL ,
	[StartFrame]         [Count] ,
	[EndFrame]           [Count] 
)
go

ALTER TABLE [WormInteraction]
	ADD CONSTRAINT [XPKWormInteraction] PRIMARY KEY  CLUSTERED ([WormInteractionKey] ASC)
go

CREATE TABLE [FeaturesPerPlateWireframe]
( 
	[FeaturesPerPlateWireframe] [Key]  NOT NULL ,
	[Value]              varbinary  NULL ,
	[PlateFeatureKey]    [Key]  NOT NULL ,
	[PlateWireframeVideoKey] [Key]  NOT NULL 
)
go

ALTER TABLE [FeaturesPerPlateWireframe]
	ADD CONSTRAINT [XPKFeaturesPerPlateWireframe] PRIMARY KEY  CLUSTERED ([FeaturesPerPlateWireframe] ASC)
go

CREATE TABLE [HistogramsPerPlateWireframe]
( 
	[HistogramsPerPlateWireframeKey] [Key]  NOT NULL ,
	[Bins]               varbinary  NULL ,
	[Counts]             varbinary  NULL ,
	[PlateWireframeVideoKey] [Key]  NOT NULL 
)
go

ALTER TABLE [HistogramsPerPlateWireframe]
	ADD CONSTRAINT [XPKHistogramsPerPlateWireframe] PRIMARY KEY  CLUSTERED ([HistogramsPerPlateWireframeKey] ASC)
go

CREATE TABLE [WormWireframeVideo]
( 
	[WormWireframeKey]   [Key]  NOT NULL ,
	[WireframeVideo]     varbinary  NULL ,
	[PlateWireframeVideoKey] [Key]  NOT NULL ,
	[DroppedFrameInfo]   varbinary  NULL 
)
go

ALTER TABLE [WormWireframeVideo]
	ADD CONSTRAINT [XPKWormWireframeVideo] PRIMARY KEY  CLUSTERED ([WormWireframeKey] ASC)
go

CREATE TABLE [HistogramsPerWormWireframe]
( 
	[HistogramsPerWormWireframeKey] [Key]  NOT NULL ,
	[Bins]               varbinary  NULL ,
	[Counts]             varbinary  NULL ,
	[SignKey]            [Key]  NOT NULL ,
	[EventDirectionKey]  [Key]  NOT NULL ,
	[WormFeatureKey]     [Key]  NOT NULL ,
	[WormWireframeKey]   [Key]  NOT NULL 
)
go

ALTER TABLE [HistogramsPerWormWireframe]
	ADD CONSTRAINT [XPKHistogramsPerWormWireframe] PRIMARY KEY  CLUSTERED ([HistogramsPerWormWireframeKey] ASC)
go

CREATE TABLE [FeaturesPerWormWireframe]
( 
	[FeaturesPerWormWireframeKey] [Key]  NOT NULL ,
	[WormFeatureKey]     [Key]  NOT NULL ,
	[Value]              varbinary  NULL ,
	[WormWireframeKey]   [Key]  NOT NULL 
)
go

ALTER TABLE [FeaturesPerWormWireframe]
	ADD CONSTRAINT [XPKFeaturesPerWormWireframe] PRIMARY KEY  CLUSTERED ([FeaturesPerWormWireframeKey] ASC)
go

CREATE TABLE [WormMeasurement]
( 
	[WormMeasurementsKey] [Key]  NOT NULL ,
	[Name]               [Name] ,
	[Description]        [Description] 
)
go

ALTER TABLE [WormMeasurement]
	ADD CONSTRAINT [XPKWormMeasurement] PRIMARY KEY  CLUSTERED ([WormMeasurementsKey] ASC)
go

CREATE TABLE [MeasurementsPerWormWireframe]
( 
	[MeasurementsPerWormWireframe] [Key]  NOT NULL ,
	[WormMeasurementsKey] [Key]  NOT NULL ,
	[Value]              varbinary  NULL ,
	[WormWireframeKey]   [Key]  NOT NULL 
)
go

ALTER TABLE [MeasurementsPerWormWireframe]
	ADD CONSTRAINT [XPKMeasurementsPerWormWireframe] PRIMARY KEY  CLUSTERED ([MeasurementsPerWormWireframe] ASC)
go


ALTER TABLE [Experimenter]
	ADD CONSTRAINT [R_48] FOREIGN KEY ([LabKey]) REFERENCES [Lab]([LabKey])
		ON DELETE NO ACTION
		ON UPDATE NO ACTION
go


ALTER TABLE [Plate]
	ADD CONSTRAINT [R_11] FOREIGN KEY ([ExperimenterKey]) REFERENCES [Experimenter]([ExperimenterKey])
		ON DELETE NO ACTION
		ON UPDATE NO ACTION
go

ALTER TABLE [Plate]
	ADD CONSTRAINT [R_46] FOREIGN KEY ([WormListKey]) REFERENCES [WormList]([WormListKey])
		ON DELETE NO ACTION
		ON UPDATE NO ACTION
go


ALTER TABLE [PlateRawVideo]
	ADD CONSTRAINT [R_13] FOREIGN KEY ([PlateKey]) REFERENCES [Plate]([PlateKey])
		ON DELETE NO ACTION
		ON UPDATE NO ACTION
go

ALTER TABLE [PlateRawVideo]
	ADD CONSTRAINT [R_28] FOREIGN KEY ([VideoMetadataKey]) REFERENCES [VideoAttributes]([VideoMetadataKey])
		ON DELETE NO ACTION
		ON UPDATE NO ACTION
go


ALTER TABLE [User]
	ADD CONSTRAINT [R_49] FOREIGN KEY ([LabKey]) REFERENCES [Lab]([LabKey])
		ON DELETE NO ACTION
		ON UPDATE NO ACTION
go


ALTER TABLE [WormFeature]
	ADD CONSTRAINT [R_5] FOREIGN KEY ([TypeKey]) REFERENCES [Type]([TypeKey])
		ON DELETE NO ACTION
		ON UPDATE NO ACTION
go

ALTER TABLE [WormFeature]
	ADD CONSTRAINT [R_6] FOREIGN KEY ([CategoryKey]) REFERENCES [Category]([CategoryKey])
		ON DELETE NO ACTION
		ON UPDATE NO ACTION
go

ALTER TABLE [WormFeature]
	ADD CONSTRAINT [R_7] FOREIGN KEY ([DirectionKey]) REFERENCES [Direction]([DirectionKey])
		ON DELETE NO ACTION
		ON UPDATE NO ACTION
go

ALTER TABLE [WormFeature]
	ADD CONSTRAINT [R_8] FOREIGN KEY ([AspectKey]) REFERENCES [Aspect]([AspectKey])
		ON DELETE NO ACTION
		ON UPDATE NO ACTION
go

ALTER TABLE [WormFeature]
	ADD CONSTRAINT [R_9] FOREIGN KEY ([BodyPartKey]) REFERENCES [BodyPart]([BodyPartKey])
		ON DELETE NO ACTION
		ON UPDATE NO ACTION
go


ALTER TABLE [Worm]
	ADD CONSTRAINT [R_1] FOREIGN KEY ([StrainKey]) REFERENCES [Strain]([StrainKey])
		ON DELETE NO ACTION
		ON UPDATE NO ACTION
go


ALTER TABLE [WormList]
	ADD CONSTRAINT [R_45] FOREIGN KEY ([WormKey]) REFERENCES [Worm]([WormKey])
		ON DELETE NO ACTION
		ON UPDATE NO ACTION
go


ALTER TABLE [PlateWireframeVideo]
	ADD CONSTRAINT [R_14] FOREIGN KEY ([PlateRawVideoKey]) REFERENCES [PlateRawVideo]([PlateRawVideoKey])
		ON DELETE NO ACTION
		ON UPDATE NO ACTION
go

ALTER TABLE [PlateWireframeVideo]
	ADD CONSTRAINT [R_15] FOREIGN KEY ([CVAlgorithmKey]) REFERENCES [ComputerVisionAlgorithm]([CVAlgorithmKey])
		ON DELETE NO ACTION
		ON UPDATE NO ACTION
go


ALTER TABLE [WormInteraction]
	ADD CONSTRAINT [R_39] FOREIGN KEY ([PlateWireframeVideoKey]) REFERENCES [PlateWireframeVideo]([PlateWireframeVideoKey])
		ON DELETE NO ACTION
		ON UPDATE NO ACTION
go

ALTER TABLE [WormInteraction]
	ADD CONSTRAINT [R_47] FOREIGN KEY ([WormListKey]) REFERENCES [WormList]([WormListKey])
		ON DELETE NO ACTION
		ON UPDATE NO ACTION
go


ALTER TABLE [FeaturesPerPlateWireframe]
	ADD CONSTRAINT [R_35] FOREIGN KEY ([PlateFeatureKey]) REFERENCES [PlateFeature]([PlateFeatureKey])
		ON DELETE NO ACTION
		ON UPDATE NO ACTION
go

ALTER TABLE [FeaturesPerPlateWireframe]
	ADD CONSTRAINT [R_36] FOREIGN KEY ([PlateWireframeVideoKey]) REFERENCES [PlateWireframeVideo]([PlateWireframeVideoKey])
		ON DELETE NO ACTION
		ON UPDATE NO ACTION
go


ALTER TABLE [HistogramsPerPlateWireframe]
	ADD CONSTRAINT [R_41] FOREIGN KEY ([PlateWireframeVideoKey]) REFERENCES [PlateWireframeVideo]([PlateWireframeVideoKey])
		ON DELETE NO ACTION
		ON UPDATE NO ACTION
go


ALTER TABLE [WormWireframeVideo]
	ADD CONSTRAINT [R_40] FOREIGN KEY ([PlateWireframeVideoKey]) REFERENCES [PlateWireframeVideo]([PlateWireframeVideoKey])
		ON DELETE NO ACTION
		ON UPDATE NO ACTION
go


ALTER TABLE [HistogramsPerWormWireframe]
	ADD CONSTRAINT [R_16] FOREIGN KEY ([SignKey]) REFERENCES [Sign]([SignKey])
		ON DELETE NO ACTION
		ON UPDATE NO ACTION
go

ALTER TABLE [HistogramsPerWormWireframe]
	ADD CONSTRAINT [R_17] FOREIGN KEY ([EventDirectionKey]) REFERENCES [Direction]([DirectionKey])
		ON DELETE NO ACTION
		ON UPDATE NO ACTION
go

ALTER TABLE [HistogramsPerWormWireframe]
	ADD CONSTRAINT [R_27] FOREIGN KEY ([WormFeatureKey]) REFERENCES [WormFeature]([WormFeatureKey])
		ON DELETE NO ACTION
		ON UPDATE NO ACTION
go

ALTER TABLE [HistogramsPerWormWireframe]
	ADD CONSTRAINT [R_38] FOREIGN KEY ([WormWireframeKey]) REFERENCES [WormWireframeVideo]([WormWireframeKey])
		ON DELETE NO ACTION
		ON UPDATE NO ACTION
go


ALTER TABLE [FeaturesPerWormWireframe]
	ADD CONSTRAINT [R_20] FOREIGN KEY ([WormFeatureKey]) REFERENCES [WormFeature]([WormFeatureKey])
		ON DELETE NO ACTION
		ON UPDATE NO ACTION
go

ALTER TABLE [FeaturesPerWormWireframe]
	ADD CONSTRAINT [R_34] FOREIGN KEY ([WormWireframeKey]) REFERENCES [WormWireframeVideo]([WormWireframeKey])
		ON DELETE NO ACTION
		ON UPDATE NO ACTION
go


ALTER TABLE [MeasurementsPerWormWireframe]
	ADD CONSTRAINT [R_23] FOREIGN KEY ([WormMeasurementsKey]) REFERENCES [WormMeasurement]([WormMeasurementsKey])
		ON DELETE NO ACTION
		ON UPDATE NO ACTION
go

ALTER TABLE [MeasurementsPerWormWireframe]
	ADD CONSTRAINT [R_37] FOREIGN KEY ([WormWireframeKey]) REFERENCES [WormWireframeVideo]([WormWireframeKey])
		ON DELETE NO ACTION
		ON UPDATE NO ACTION
go


CREATE TRIGGER tD_Aspect ON Aspect FOR DELETE AS
/* ERwin Builtin Trigger */
/* DELETE trigger on Aspect */
BEGIN
  DECLARE  @errno   int,
           @errmsg  varchar(255)
    /* ERwin Builtin Trigger */
    /* Aspect  WormFeature on parent delete no action */
    /* ERWIN_RELATION:CHECKSUM="0000f5b6", PARENT_OWNER="", PARENT_TABLE="Aspect"
    CHILD_OWNER="", CHILD_TABLE="WormFeature"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_8", FK_COLUMNS="AspectKey" */
    IF EXISTS (
      SELECT * FROM deleted,WormFeature
      WHERE
        /*  %JoinFKPK(WormFeature,deleted," = "," AND") */
        WormFeature.AspectKey = deleted.AspectKey
    )
    BEGIN
      SELECT @errno  = 30001,
             @errmsg = 'Cannot delete Aspect because WormFeature exists.'
      GOTO error
    END


    /* ERwin Builtin Trigger */
    RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 

CREATE TRIGGER tU_Aspect ON Aspect FOR UPDATE AS
/* ERwin Builtin Trigger */
/* UPDATE trigger on Aspect */
BEGIN
  DECLARE  @numrows int,
           @nullcnt int,
           @validcnt int,
           @insAspectKey Key,
           @errno   int,
           @errmsg  varchar(255)

  SELECT @numrows = @@rowcount
  /* ERwin Builtin Trigger */
  /* Aspect  WormFeature on parent update no action */
  /* ERWIN_RELATION:CHECKSUM="00010ba8", PARENT_OWNER="", PARENT_TABLE="Aspect"
    CHILD_OWNER="", CHILD_TABLE="WormFeature"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_8", FK_COLUMNS="AspectKey" */
  IF
    /* %ParentPK(" OR",UPDATE) */
    UPDATE(AspectKey)
  BEGIN
    IF EXISTS (
      SELECT * FROM deleted,WormFeature
      WHERE
        /*  %JoinFKPK(WormFeature,deleted," = "," AND") */
        WormFeature.AspectKey = deleted.AspectKey
    )
    BEGIN
      SELECT @errno  = 30005,
             @errmsg = 'Cannot update Aspect because WormFeature exists.'
      GOTO error
    END
  END


  /* ERwin Builtin Trigger */
  RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 



CREATE TRIGGER tD_BodyPart ON BodyPart FOR DELETE AS
/* ERwin Builtin Trigger */
/* DELETE trigger on BodyPart */
BEGIN
  DECLARE  @errno   int,
           @errmsg  varchar(255)
    /* ERwin Builtin Trigger */
    /* BodyPart  WormFeature on parent delete no action */
    /* ERWIN_RELATION:CHECKSUM="0000fc74", PARENT_OWNER="", PARENT_TABLE="BodyPart"
    CHILD_OWNER="", CHILD_TABLE="WormFeature"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_9", FK_COLUMNS="BodyPartKey" */
    IF EXISTS (
      SELECT * FROM deleted,WormFeature
      WHERE
        /*  %JoinFKPK(WormFeature,deleted," = "," AND") */
        WormFeature.BodyPartKey = deleted.BodyPartKey
    )
    BEGIN
      SELECT @errno  = 30001,
             @errmsg = 'Cannot delete BodyPart because WormFeature exists.'
      GOTO error
    END


    /* ERwin Builtin Trigger */
    RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 

CREATE TRIGGER tU_BodyPart ON BodyPart FOR UPDATE AS
/* ERwin Builtin Trigger */
/* UPDATE trigger on BodyPart */
BEGIN
  DECLARE  @numrows int,
           @nullcnt int,
           @validcnt int,
           @insBodyPartKey Key,
           @errno   int,
           @errmsg  varchar(255)

  SELECT @numrows = @@rowcount
  /* ERwin Builtin Trigger */
  /* BodyPart  WormFeature on parent update no action */
  /* ERWIN_RELATION:CHECKSUM="000108d7", PARENT_OWNER="", PARENT_TABLE="BodyPart"
    CHILD_OWNER="", CHILD_TABLE="WormFeature"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_9", FK_COLUMNS="BodyPartKey" */
  IF
    /* %ParentPK(" OR",UPDATE) */
    UPDATE(BodyPartKey)
  BEGIN
    IF EXISTS (
      SELECT * FROM deleted,WormFeature
      WHERE
        /*  %JoinFKPK(WormFeature,deleted," = "," AND") */
        WormFeature.BodyPartKey = deleted.BodyPartKey
    )
    BEGIN
      SELECT @errno  = 30005,
             @errmsg = 'Cannot update BodyPart because WormFeature exists.'
      GOTO error
    END
  END


  /* ERwin Builtin Trigger */
  RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 



CREATE TRIGGER tD_Category ON Category FOR DELETE AS
/* ERwin Builtin Trigger */
/* DELETE trigger on Category */
BEGIN
  DECLARE  @errno   int,
           @errmsg  varchar(255)
    /* ERwin Builtin Trigger */
    /* Category  WormFeature on parent delete no action */
    /* ERWIN_RELATION:CHECKSUM="0000f5bb", PARENT_OWNER="", PARENT_TABLE="Category"
    CHILD_OWNER="", CHILD_TABLE="WormFeature"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_6", FK_COLUMNS="CategoryKey" */
    IF EXISTS (
      SELECT * FROM deleted,WormFeature
      WHERE
        /*  %JoinFKPK(WormFeature,deleted," = "," AND") */
        WormFeature.CategoryKey = deleted.CategoryKey
    )
    BEGIN
      SELECT @errno  = 30001,
             @errmsg = 'Cannot delete Category because WormFeature exists.'
      GOTO error
    END


    /* ERwin Builtin Trigger */
    RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 

CREATE TRIGGER tU_Category ON Category FOR UPDATE AS
/* ERwin Builtin Trigger */
/* UPDATE trigger on Category */
BEGIN
  DECLARE  @numrows int,
           @nullcnt int,
           @validcnt int,
           @insCategoryKey Key,
           @errno   int,
           @errmsg  varchar(255)

  SELECT @numrows = @@rowcount
  /* ERwin Builtin Trigger */
  /* Category  WormFeature on parent update no action */
  /* ERWIN_RELATION:CHECKSUM="00010f18", PARENT_OWNER="", PARENT_TABLE="Category"
    CHILD_OWNER="", CHILD_TABLE="WormFeature"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_6", FK_COLUMNS="CategoryKey" */
  IF
    /* %ParentPK(" OR",UPDATE) */
    UPDATE(CategoryKey)
  BEGIN
    IF EXISTS (
      SELECT * FROM deleted,WormFeature
      WHERE
        /*  %JoinFKPK(WormFeature,deleted," = "," AND") */
        WormFeature.CategoryKey = deleted.CategoryKey
    )
    BEGIN
      SELECT @errno  = 30005,
             @errmsg = 'Cannot update Category because WormFeature exists.'
      GOTO error
    END
  END


  /* ERwin Builtin Trigger */
  RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 



CREATE TRIGGER tD_ComputerVisionAlgorithm ON ComputerVisionAlgorithm FOR DELETE AS
/* ERwin Builtin Trigger */
/* DELETE trigger on ComputerVisionAlgorithm */
BEGIN
  DECLARE  @errno   int,
           @errmsg  varchar(255)
    /* ERwin Builtin Trigger */
    /* ComputerVisionAlgorithm  PlateWireframeVideo on parent delete no action */
    /* ERWIN_RELATION:CHECKSUM="0001268f", PARENT_OWNER="", PARENT_TABLE="ComputerVisionAlgorithm"
    CHILD_OWNER="", CHILD_TABLE="PlateWireframeVideo"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_15", FK_COLUMNS="CVAlgorithmKey" */
    IF EXISTS (
      SELECT * FROM deleted,PlateWireframeVideo
      WHERE
        /*  %JoinFKPK(PlateWireframeVideo,deleted," = "," AND") */
        PlateWireframeVideo.CVAlgorithmKey = deleted.CVAlgorithmKey
    )
    BEGIN
      SELECT @errno  = 30001,
             @errmsg = 'Cannot delete ComputerVisionAlgorithm because PlateWireframeVideo exists.'
      GOTO error
    END


    /* ERwin Builtin Trigger */
    RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 

CREATE TRIGGER tU_ComputerVisionAlgorithm ON ComputerVisionAlgorithm FOR UPDATE AS
/* ERwin Builtin Trigger */
/* UPDATE trigger on ComputerVisionAlgorithm */
BEGIN
  DECLARE  @numrows int,
           @nullcnt int,
           @validcnt int,
           @insCVAlgorithmKey Key,
           @errno   int,
           @errmsg  varchar(255)

  SELECT @numrows = @@rowcount
  /* ERwin Builtin Trigger */
  /* ComputerVisionAlgorithm  PlateWireframeVideo on parent update no action */
  /* ERWIN_RELATION:CHECKSUM="00013e9d", PARENT_OWNER="", PARENT_TABLE="ComputerVisionAlgorithm"
    CHILD_OWNER="", CHILD_TABLE="PlateWireframeVideo"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_15", FK_COLUMNS="CVAlgorithmKey" */
  IF
    /* %ParentPK(" OR",UPDATE) */
    UPDATE(CVAlgorithmKey)
  BEGIN
    IF EXISTS (
      SELECT * FROM deleted,PlateWireframeVideo
      WHERE
        /*  %JoinFKPK(PlateWireframeVideo,deleted," = "," AND") */
        PlateWireframeVideo.CVAlgorithmKey = deleted.CVAlgorithmKey
    )
    BEGIN
      SELECT @errno  = 30005,
             @errmsg = 'Cannot update ComputerVisionAlgorithm because PlateWireframeVideo exists.'
      GOTO error
    END
  END


  /* ERwin Builtin Trigger */
  RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 



CREATE TRIGGER tD_Direction ON Direction FOR DELETE AS
/* ERwin Builtin Trigger */
/* DELETE trigger on Direction */
BEGIN
  DECLARE  @errno   int,
           @errmsg  varchar(255)
    /* ERwin Builtin Trigger */
    /* Direction  HistogramsPerWormWireframe on parent delete no action */
    /* ERWIN_RELATION:CHECKSUM="00021fa3", PARENT_OWNER="", PARENT_TABLE="Direction"
    CHILD_OWNER="", CHILD_TABLE="HistogramsPerWormWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_17", FK_COLUMNS="EventDirectionKey" */
    IF EXISTS (
      SELECT * FROM deleted,HistogramsPerWormWireframe
      WHERE
        /*  %JoinFKPK(HistogramsPerWormWireframe,deleted," = "," AND") */
        HistogramsPerWormWireframe.EventDirectionKey = deleted.DirectionKey
    )
    BEGIN
      SELECT @errno  = 30001,
             @errmsg = 'Cannot delete Direction because HistogramsPerWormWireframe exists.'
      GOTO error
    END

    /* ERwin Builtin Trigger */
    /* Direction  WormFeature on parent delete no action */
    /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="Direction"
    CHILD_OWNER="", CHILD_TABLE="WormFeature"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_7", FK_COLUMNS="DirectionKey" */
    IF EXISTS (
      SELECT * FROM deleted,WormFeature
      WHERE
        /*  %JoinFKPK(WormFeature,deleted," = "," AND") */
        WormFeature.DirectionKey = deleted.DirectionKey
    )
    BEGIN
      SELECT @errno  = 30001,
             @errmsg = 'Cannot delete Direction because WormFeature exists.'
      GOTO error
    END


    /* ERwin Builtin Trigger */
    RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 

CREATE TRIGGER tU_Direction ON Direction FOR UPDATE AS
/* ERwin Builtin Trigger */
/* UPDATE trigger on Direction */
BEGIN
  DECLARE  @numrows int,
           @nullcnt int,
           @validcnt int,
           @insDirectionKey Key,
           @errno   int,
           @errmsg  varchar(255)

  SELECT @numrows = @@rowcount
  /* ERwin Builtin Trigger */
  /* Direction  HistogramsPerWormWireframe on parent update no action */
  /* ERWIN_RELATION:CHECKSUM="00024ceb", PARENT_OWNER="", PARENT_TABLE="Direction"
    CHILD_OWNER="", CHILD_TABLE="HistogramsPerWormWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_17", FK_COLUMNS="EventDirectionKey" */
  IF
    /* %ParentPK(" OR",UPDATE) */
    UPDATE(DirectionKey)
  BEGIN
    IF EXISTS (
      SELECT * FROM deleted,HistogramsPerWormWireframe
      WHERE
        /*  %JoinFKPK(HistogramsPerWormWireframe,deleted," = "," AND") */
        HistogramsPerWormWireframe.EventDirectionKey = deleted.DirectionKey
    )
    BEGIN
      SELECT @errno  = 30005,
             @errmsg = 'Cannot update Direction because HistogramsPerWormWireframe exists.'
      GOTO error
    END
  END

  /* ERwin Builtin Trigger */
  /* Direction  WormFeature on parent update no action */
  /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="Direction"
    CHILD_OWNER="", CHILD_TABLE="WormFeature"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_7", FK_COLUMNS="DirectionKey" */
  IF
    /* %ParentPK(" OR",UPDATE) */
    UPDATE(DirectionKey)
  BEGIN
    IF EXISTS (
      SELECT * FROM deleted,WormFeature
      WHERE
        /*  %JoinFKPK(WormFeature,deleted," = "," AND") */
        WormFeature.DirectionKey = deleted.DirectionKey
    )
    BEGIN
      SELECT @errno  = 30005,
             @errmsg = 'Cannot update Direction because WormFeature exists.'
      GOTO error
    END
  END


  /* ERwin Builtin Trigger */
  RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 



CREATE TRIGGER tD_Experimenter ON Experimenter FOR DELETE AS
/* ERwin Builtin Trigger */
/* DELETE trigger on Experimenter */
BEGIN
  DECLARE  @errno   int,
           @errmsg  varchar(255)
    /* ERwin Builtin Trigger */
    /* Experimenter  Plate on parent delete no action */
    /* ERWIN_RELATION:CHECKSUM="00020061", PARENT_OWNER="", PARENT_TABLE="Experimenter"
    CHILD_OWNER="", CHILD_TABLE="Plate"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_11", FK_COLUMNS="ExperimenterKey" */
    IF EXISTS (
      SELECT * FROM deleted,Plate
      WHERE
        /*  %JoinFKPK(Plate,deleted," = "," AND") */
        Plate.ExperimenterKey = deleted.ExperimenterKey
    )
    BEGIN
      SELECT @errno  = 30001,
             @errmsg = 'Cannot delete Experimenter because Plate exists.'
      GOTO error
    END

    /* ERwin Builtin Trigger */
    /* Lab  Experimenter on child delete no action */
    /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="Lab"
    CHILD_OWNER="", CHILD_TABLE="Experimenter"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_48", FK_COLUMNS="LabKey" */
    IF EXISTS (SELECT * FROM deleted,Lab
      WHERE
        /* %JoinFKPK(deleted,Lab," = "," AND") */
        deleted.LabKey = Lab.LabKey AND
        NOT EXISTS (
          SELECT * FROM Experimenter
          WHERE
            /* %JoinFKPK(Experimenter,Lab," = "," AND") */
            Experimenter.LabKey = Lab.LabKey
        )
    )
    BEGIN
      SELECT @errno  = 30010,
             @errmsg = 'Cannot delete last Experimenter because Lab exists.'
      GOTO error
    END


    /* ERwin Builtin Trigger */
    RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 

CREATE TRIGGER tU_Experimenter ON Experimenter FOR UPDATE AS
/* ERwin Builtin Trigger */
/* UPDATE trigger on Experimenter */
BEGIN
  DECLARE  @numrows int,
           @nullcnt int,
           @validcnt int,
           @insExperimenterKey Key,
           @errno   int,
           @errmsg  varchar(255)

  SELECT @numrows = @@rowcount
  /* ERwin Builtin Trigger */
  /* Experimenter  Plate on parent update no action */
  /* ERWIN_RELATION:CHECKSUM="00023cf4", PARENT_OWNER="", PARENT_TABLE="Experimenter"
    CHILD_OWNER="", CHILD_TABLE="Plate"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_11", FK_COLUMNS="ExperimenterKey" */
  IF
    /* %ParentPK(" OR",UPDATE) */
    UPDATE(ExperimenterKey)
  BEGIN
    IF EXISTS (
      SELECT * FROM deleted,Plate
      WHERE
        /*  %JoinFKPK(Plate,deleted," = "," AND") */
        Plate.ExperimenterKey = deleted.ExperimenterKey
    )
    BEGIN
      SELECT @errno  = 30005,
             @errmsg = 'Cannot update Experimenter because Plate exists.'
      GOTO error
    END
  END

  /* ERwin Builtin Trigger */
  /* Lab  Experimenter on child update no action */
  /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="Lab"
    CHILD_OWNER="", CHILD_TABLE="Experimenter"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_48", FK_COLUMNS="LabKey" */
  IF
    /* %ChildFK(" OR",UPDATE) */
    UPDATE(LabKey)
  BEGIN
    SELECT @nullcnt = 0
    SELECT @validcnt = count(*)
      FROM inserted,Lab
        WHERE
          /* %JoinFKPK(inserted,Lab) */
          inserted.LabKey = Lab.LabKey
    /* %NotnullFK(inserted," IS NULL","select @nullcnt = count(*) from inserted where"," AND") */
    
    IF @validcnt + @nullcnt != @numrows
    BEGIN
      SELECT @errno  = 30007,
             @errmsg = 'Cannot update Experimenter because Lab does not exist.'
      GOTO error
    END
  END


  /* ERwin Builtin Trigger */
  RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 



CREATE TRIGGER tD_Lab ON Lab FOR DELETE AS
/* ERwin Builtin Trigger */
/* DELETE trigger on Lab */
BEGIN
  DECLARE  @errno   int,
           @errmsg  varchar(255)
    /* ERwin Builtin Trigger */
    /* Lab  User on parent delete no action */
    /* ERWIN_RELATION:CHECKSUM="0001c405", PARENT_OWNER="", PARENT_TABLE="Lab"
    CHILD_OWNER="", CHILD_TABLE="User"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_49", FK_COLUMNS="LabKey" */
    IF EXISTS (
      SELECT * FROM deleted,User
      WHERE
        /*  %JoinFKPK(User,deleted," = "," AND") */
        User.LabKey = deleted.LabKey
    )
    BEGIN
      SELECT @errno  = 30001,
             @errmsg = 'Cannot delete Lab because User exists.'
      GOTO error
    END

    /* ERwin Builtin Trigger */
    /* Lab  Experimenter on parent delete no action */
    /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="Lab"
    CHILD_OWNER="", CHILD_TABLE="Experimenter"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_48", FK_COLUMNS="LabKey" */
    IF EXISTS (
      SELECT * FROM deleted,Experimenter
      WHERE
        /*  %JoinFKPK(Experimenter,deleted," = "," AND") */
        Experimenter.LabKey = deleted.LabKey
    )
    BEGIN
      SELECT @errno  = 30001,
             @errmsg = 'Cannot delete Lab because Experimenter exists.'
      GOTO error
    END


    /* ERwin Builtin Trigger */
    RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 

CREATE TRIGGER tU_Lab ON Lab FOR UPDATE AS
/* ERwin Builtin Trigger */
/* UPDATE trigger on Lab */
BEGIN
  DECLARE  @numrows int,
           @nullcnt int,
           @validcnt int,
           @insLabKey Key,
           @errno   int,
           @errmsg  varchar(255)

  SELECT @numrows = @@rowcount
  /* ERwin Builtin Trigger */
  /* Lab  User on parent update no action */
  /* ERWIN_RELATION:CHECKSUM="0001efe3", PARENT_OWNER="", PARENT_TABLE="Lab"
    CHILD_OWNER="", CHILD_TABLE="User"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_49", FK_COLUMNS="LabKey" */
  IF
    /* %ParentPK(" OR",UPDATE) */
    UPDATE(LabKey)
  BEGIN
    IF EXISTS (
      SELECT * FROM deleted,User
      WHERE
        /*  %JoinFKPK(User,deleted," = "," AND") */
        User.LabKey = deleted.LabKey
    )
    BEGIN
      SELECT @errno  = 30005,
             @errmsg = 'Cannot update Lab because User exists.'
      GOTO error
    END
  END

  /* ERwin Builtin Trigger */
  /* Lab  Experimenter on parent update no action */
  /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="Lab"
    CHILD_OWNER="", CHILD_TABLE="Experimenter"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_48", FK_COLUMNS="LabKey" */
  IF
    /* %ParentPK(" OR",UPDATE) */
    UPDATE(LabKey)
  BEGIN
    IF EXISTS (
      SELECT * FROM deleted,Experimenter
      WHERE
        /*  %JoinFKPK(Experimenter,deleted," = "," AND") */
        Experimenter.LabKey = deleted.LabKey
    )
    BEGIN
      SELECT @errno  = 30005,
             @errmsg = 'Cannot update Lab because Experimenter exists.'
      GOTO error
    END
  END


  /* ERwin Builtin Trigger */
  RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 



CREATE TRIGGER tD_Plate ON Plate FOR DELETE AS
/* ERwin Builtin Trigger */
/* DELETE trigger on Plate */
BEGIN
  DECLARE  @errno   int,
           @errmsg  varchar(255)
    /* ERwin Builtin Trigger */
    /* Plate  PlateRawVideo on parent delete no action */
    /* ERWIN_RELATION:CHECKSUM="00035db6", PARENT_OWNER="", PARENT_TABLE="Plate"
    CHILD_OWNER="", CHILD_TABLE="PlateRawVideo"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_13", FK_COLUMNS="PlateKey" */
    IF EXISTS (
      SELECT * FROM deleted,PlateRawVideo
      WHERE
        /*  %JoinFKPK(PlateRawVideo,deleted," = "," AND") */
        PlateRawVideo.PlateKey = deleted.PlateKey
    )
    BEGIN
      SELECT @errno  = 30001,
             @errmsg = 'Cannot delete Plate because PlateRawVideo exists.'
      GOTO error
    END

    /* ERwin Builtin Trigger */
    /* WormList  Plate on child delete no action */
    /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="WormList"
    CHILD_OWNER="", CHILD_TABLE="Plate"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_46", FK_COLUMNS="WormListKey" */
    IF EXISTS (SELECT * FROM deleted,WormList
      WHERE
        /* %JoinFKPK(deleted,WormList," = "," AND") */
        deleted.WormListKey = WormList.WormListKey AND
        NOT EXISTS (
          SELECT * FROM Plate
          WHERE
            /* %JoinFKPK(Plate,WormList," = "," AND") */
            Plate.WormListKey = WormList.WormListKey
        )
    )
    BEGIN
      SELECT @errno  = 30010,
             @errmsg = 'Cannot delete last Plate because WormList exists.'
      GOTO error
    END

    /* ERwin Builtin Trigger */
    /* Experimenter  Plate on child delete no action */
    /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="Experimenter"
    CHILD_OWNER="", CHILD_TABLE="Plate"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_11", FK_COLUMNS="ExperimenterKey" */
    IF EXISTS (SELECT * FROM deleted,Experimenter
      WHERE
        /* %JoinFKPK(deleted,Experimenter," = "," AND") */
        deleted.ExperimenterKey = Experimenter.ExperimenterKey AND
        NOT EXISTS (
          SELECT * FROM Plate
          WHERE
            /* %JoinFKPK(Plate,Experimenter," = "," AND") */
            Plate.ExperimenterKey = Experimenter.ExperimenterKey
        )
    )
    BEGIN
      SELECT @errno  = 30010,
             @errmsg = 'Cannot delete last Plate because Experimenter exists.'
      GOTO error
    END


    /* ERwin Builtin Trigger */
    RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 

CREATE TRIGGER tU_Plate ON Plate FOR UPDATE AS
/* ERwin Builtin Trigger */
/* UPDATE trigger on Plate */
BEGIN
  DECLARE  @numrows int,
           @nullcnt int,
           @validcnt int,
           @insPlateKey Key,
           @errno   int,
           @errmsg  varchar(255)

  SELECT @numrows = @@rowcount
  /* ERwin Builtin Trigger */
  /* Plate  PlateRawVideo on parent update no action */
  /* ERWIN_RELATION:CHECKSUM="0003a6c8", PARENT_OWNER="", PARENT_TABLE="Plate"
    CHILD_OWNER="", CHILD_TABLE="PlateRawVideo"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_13", FK_COLUMNS="PlateKey" */
  IF
    /* %ParentPK(" OR",UPDATE) */
    UPDATE(PlateKey)
  BEGIN
    IF EXISTS (
      SELECT * FROM deleted,PlateRawVideo
      WHERE
        /*  %JoinFKPK(PlateRawVideo,deleted," = "," AND") */
        PlateRawVideo.PlateKey = deleted.PlateKey
    )
    BEGIN
      SELECT @errno  = 30005,
             @errmsg = 'Cannot update Plate because PlateRawVideo exists.'
      GOTO error
    END
  END

  /* ERwin Builtin Trigger */
  /* WormList  Plate on child update no action */
  /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="WormList"
    CHILD_OWNER="", CHILD_TABLE="Plate"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_46", FK_COLUMNS="WormListKey" */
  IF
    /* %ChildFK(" OR",UPDATE) */
    UPDATE(WormListKey)
  BEGIN
    SELECT @nullcnt = 0
    SELECT @validcnt = count(*)
      FROM inserted,WormList
        WHERE
          /* %JoinFKPK(inserted,WormList) */
          inserted.WormListKey = WormList.WormListKey
    /* %NotnullFK(inserted," IS NULL","select @nullcnt = count(*) from inserted where"," AND") */
    
    IF @validcnt + @nullcnt != @numrows
    BEGIN
      SELECT @errno  = 30007,
             @errmsg = 'Cannot update Plate because WormList does not exist.'
      GOTO error
    END
  END

  /* ERwin Builtin Trigger */
  /* Experimenter  Plate on child update no action */
  /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="Experimenter"
    CHILD_OWNER="", CHILD_TABLE="Plate"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_11", FK_COLUMNS="ExperimenterKey" */
  IF
    /* %ChildFK(" OR",UPDATE) */
    UPDATE(ExperimenterKey)
  BEGIN
    SELECT @nullcnt = 0
    SELECT @validcnt = count(*)
      FROM inserted,Experimenter
        WHERE
          /* %JoinFKPK(inserted,Experimenter) */
          inserted.ExperimenterKey = Experimenter.ExperimenterKey
    /* %NotnullFK(inserted," IS NULL","select @nullcnt = count(*) from inserted where"," AND") */
    
    IF @validcnt + @nullcnt != @numrows
    BEGIN
      SELECT @errno  = 30007,
             @errmsg = 'Cannot update Plate because Experimenter does not exist.'
      GOTO error
    END
  END


  /* ERwin Builtin Trigger */
  RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 



CREATE TRIGGER tD_PlateFeature ON PlateFeature FOR DELETE AS
/* ERwin Builtin Trigger */
/* DELETE trigger on PlateFeature */
BEGIN
  DECLARE  @errno   int,
           @errmsg  varchar(255)
    /* ERwin Builtin Trigger */
    /* PlateFeature  FeaturesPerPlateWireframe on parent delete no action */
    /* ERWIN_RELATION:CHECKSUM="00012901", PARENT_OWNER="", PARENT_TABLE="PlateFeature"
    CHILD_OWNER="", CHILD_TABLE="FeaturesPerPlateWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_35", FK_COLUMNS="PlateFeatureKey" */
    IF EXISTS (
      SELECT * FROM deleted,FeaturesPerPlateWireframe
      WHERE
        /*  %JoinFKPK(FeaturesPerPlateWireframe,deleted," = "," AND") */
        FeaturesPerPlateWireframe.PlateFeatureKey = deleted.PlateFeatureKey
    )
    BEGIN
      SELECT @errno  = 30001,
             @errmsg = 'Cannot delete PlateFeature because FeaturesPerPlateWireframe exists.'
      GOTO error
    END


    /* ERwin Builtin Trigger */
    RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 

CREATE TRIGGER tU_PlateFeature ON PlateFeature FOR UPDATE AS
/* ERwin Builtin Trigger */
/* UPDATE trigger on PlateFeature */
BEGIN
  DECLARE  @numrows int,
           @nullcnt int,
           @validcnt int,
           @insPlateFeatureKey Key,
           @errno   int,
           @errmsg  varchar(255)

  SELECT @numrows = @@rowcount
  /* ERwin Builtin Trigger */
  /* PlateFeature  FeaturesPerPlateWireframe on parent update no action */
  /* ERWIN_RELATION:CHECKSUM="00014e7f", PARENT_OWNER="", PARENT_TABLE="PlateFeature"
    CHILD_OWNER="", CHILD_TABLE="FeaturesPerPlateWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_35", FK_COLUMNS="PlateFeatureKey" */
  IF
    /* %ParentPK(" OR",UPDATE) */
    UPDATE(PlateFeatureKey)
  BEGIN
    IF EXISTS (
      SELECT * FROM deleted,FeaturesPerPlateWireframe
      WHERE
        /*  %JoinFKPK(FeaturesPerPlateWireframe,deleted," = "," AND") */
        FeaturesPerPlateWireframe.PlateFeatureKey = deleted.PlateFeatureKey
    )
    BEGIN
      SELECT @errno  = 30005,
             @errmsg = 'Cannot update PlateFeature because FeaturesPerPlateWireframe exists.'
      GOTO error
    END
  END


  /* ERwin Builtin Trigger */
  RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 



CREATE TRIGGER tD_PlateRawVideo ON PlateRawVideo FOR DELETE AS
/* ERwin Builtin Trigger */
/* DELETE trigger on PlateRawVideo */
BEGIN
  DECLARE  @errno   int,
           @errmsg  varchar(255)
    /* ERwin Builtin Trigger */
    /* PlateRawVideo  PlateWireframeVideo on parent delete no action */
    /* ERWIN_RELATION:CHECKSUM="0003a0e5", PARENT_OWNER="", PARENT_TABLE="PlateRawVideo"
    CHILD_OWNER="", CHILD_TABLE="PlateWireframeVideo"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_14", FK_COLUMNS="PlateRawVideoKey" */
    IF EXISTS (
      SELECT * FROM deleted,PlateWireframeVideo
      WHERE
        /*  %JoinFKPK(PlateWireframeVideo,deleted," = "," AND") */
        PlateWireframeVideo.PlateRawVideoKey = deleted.PlateRawVideoKey
    )
    BEGIN
      SELECT @errno  = 30001,
             @errmsg = 'Cannot delete PlateRawVideo because PlateWireframeVideo exists.'
      GOTO error
    END

    /* ERwin Builtin Trigger */
    /* VideoAttributes  PlateRawVideo on child delete no action */
    /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="VideoAttributes"
    CHILD_OWNER="", CHILD_TABLE="PlateRawVideo"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_28", FK_COLUMNS="VideoMetadataKey" */
    IF EXISTS (SELECT * FROM deleted,VideoAttributes
      WHERE
        /* %JoinFKPK(deleted,VideoAttributes," = "," AND") */
        deleted.VideoMetadataKey = VideoAttributes.VideoMetadataKey AND
        NOT EXISTS (
          SELECT * FROM PlateRawVideo
          WHERE
            /* %JoinFKPK(PlateRawVideo,VideoAttributes," = "," AND") */
            PlateRawVideo.VideoMetadataKey = VideoAttributes.VideoMetadataKey
        )
    )
    BEGIN
      SELECT @errno  = 30010,
             @errmsg = 'Cannot delete last PlateRawVideo because VideoAttributes exists.'
      GOTO error
    END

    /* ERwin Builtin Trigger */
    /* Plate  PlateRawVideo on child delete no action */
    /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="Plate"
    CHILD_OWNER="", CHILD_TABLE="PlateRawVideo"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_13", FK_COLUMNS="PlateKey" */
    IF EXISTS (SELECT * FROM deleted,Plate
      WHERE
        /* %JoinFKPK(deleted,Plate," = "," AND") */
        deleted.PlateKey = Plate.PlateKey AND
        NOT EXISTS (
          SELECT * FROM PlateRawVideo
          WHERE
            /* %JoinFKPK(PlateRawVideo,Plate," = "," AND") */
            PlateRawVideo.PlateKey = Plate.PlateKey
        )
    )
    BEGIN
      SELECT @errno  = 30010,
             @errmsg = 'Cannot delete last PlateRawVideo because Plate exists.'
      GOTO error
    END


    /* ERwin Builtin Trigger */
    RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 

CREATE TRIGGER tU_PlateRawVideo ON PlateRawVideo FOR UPDATE AS
/* ERwin Builtin Trigger */
/* UPDATE trigger on PlateRawVideo */
BEGIN
  DECLARE  @numrows int,
           @nullcnt int,
           @validcnt int,
           @insPlateRawVideoKey Key,
           @errno   int,
           @errmsg  varchar(255)

  SELECT @numrows = @@rowcount
  /* ERwin Builtin Trigger */
  /* PlateRawVideo  PlateWireframeVideo on parent update no action */
  /* ERWIN_RELATION:CHECKSUM="0003ee15", PARENT_OWNER="", PARENT_TABLE="PlateRawVideo"
    CHILD_OWNER="", CHILD_TABLE="PlateWireframeVideo"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_14", FK_COLUMNS="PlateRawVideoKey" */
  IF
    /* %ParentPK(" OR",UPDATE) */
    UPDATE(PlateRawVideoKey)
  BEGIN
    IF EXISTS (
      SELECT * FROM deleted,PlateWireframeVideo
      WHERE
        /*  %JoinFKPK(PlateWireframeVideo,deleted," = "," AND") */
        PlateWireframeVideo.PlateRawVideoKey = deleted.PlateRawVideoKey
    )
    BEGIN
      SELECT @errno  = 30005,
             @errmsg = 'Cannot update PlateRawVideo because PlateWireframeVideo exists.'
      GOTO error
    END
  END

  /* ERwin Builtin Trigger */
  /* VideoAttributes  PlateRawVideo on child update no action */
  /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="VideoAttributes"
    CHILD_OWNER="", CHILD_TABLE="PlateRawVideo"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_28", FK_COLUMNS="VideoMetadataKey" */
  IF
    /* %ChildFK(" OR",UPDATE) */
    UPDATE(VideoMetadataKey)
  BEGIN
    SELECT @nullcnt = 0
    SELECT @validcnt = count(*)
      FROM inserted,VideoAttributes
        WHERE
          /* %JoinFKPK(inserted,VideoAttributes) */
          inserted.VideoMetadataKey = VideoAttributes.VideoMetadataKey
    /* %NotnullFK(inserted," IS NULL","select @nullcnt = count(*) from inserted where"," AND") */
    
    IF @validcnt + @nullcnt != @numrows
    BEGIN
      SELECT @errno  = 30007,
             @errmsg = 'Cannot update PlateRawVideo because VideoAttributes does not exist.'
      GOTO error
    END
  END

  /* ERwin Builtin Trigger */
  /* Plate  PlateRawVideo on child update no action */
  /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="Plate"
    CHILD_OWNER="", CHILD_TABLE="PlateRawVideo"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_13", FK_COLUMNS="PlateKey" */
  IF
    /* %ChildFK(" OR",UPDATE) */
    UPDATE(PlateKey)
  BEGIN
    SELECT @nullcnt = 0
    SELECT @validcnt = count(*)
      FROM inserted,Plate
        WHERE
          /* %JoinFKPK(inserted,Plate) */
          inserted.PlateKey = Plate.PlateKey
    /* %NotnullFK(inserted," IS NULL","select @nullcnt = count(*) from inserted where"," AND") */
    
    IF @validcnt + @nullcnt != @numrows
    BEGIN
      SELECT @errno  = 30007,
             @errmsg = 'Cannot update PlateRawVideo because Plate does not exist.'
      GOTO error
    END
  END


  /* ERwin Builtin Trigger */
  RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 



CREATE TRIGGER tD_Sign ON Sign FOR DELETE AS
/* ERwin Builtin Trigger */
/* DELETE trigger on Sign */
BEGIN
  DECLARE  @errno   int,
           @errmsg  varchar(255)
    /* ERwin Builtin Trigger */
    /* Sign  HistogramsPerWormWireframe on parent delete no action */
    /* ERWIN_RELATION:CHECKSUM="00010fc8", PARENT_OWNER="", PARENT_TABLE="Sign"
    CHILD_OWNER="", CHILD_TABLE="HistogramsPerWormWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_16", FK_COLUMNS="SignKey" */
    IF EXISTS (
      SELECT * FROM deleted,HistogramsPerWormWireframe
      WHERE
        /*  %JoinFKPK(HistogramsPerWormWireframe,deleted," = "," AND") */
        HistogramsPerWormWireframe.SignKey = deleted.SignKey
    )
    BEGIN
      SELECT @errno  = 30001,
             @errmsg = 'Cannot delete Sign because HistogramsPerWormWireframe exists.'
      GOTO error
    END


    /* ERwin Builtin Trigger */
    RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 

CREATE TRIGGER tU_Sign ON Sign FOR UPDATE AS
/* ERwin Builtin Trigger */
/* UPDATE trigger on Sign */
BEGIN
  DECLARE  @numrows int,
           @nullcnt int,
           @validcnt int,
           @insSignKey Key,
           @errno   int,
           @errmsg  varchar(255)

  SELECT @numrows = @@rowcount
  /* ERwin Builtin Trigger */
  /* Sign  HistogramsPerWormWireframe on parent update no action */
  /* ERWIN_RELATION:CHECKSUM="00012e0e", PARENT_OWNER="", PARENT_TABLE="Sign"
    CHILD_OWNER="", CHILD_TABLE="HistogramsPerWormWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_16", FK_COLUMNS="SignKey" */
  IF
    /* %ParentPK(" OR",UPDATE) */
    UPDATE(SignKey)
  BEGIN
    IF EXISTS (
      SELECT * FROM deleted,HistogramsPerWormWireframe
      WHERE
        /*  %JoinFKPK(HistogramsPerWormWireframe,deleted," = "," AND") */
        HistogramsPerWormWireframe.SignKey = deleted.SignKey
    )
    BEGIN
      SELECT @errno  = 30005,
             @errmsg = 'Cannot update Sign because HistogramsPerWormWireframe exists.'
      GOTO error
    END
  END


  /* ERwin Builtin Trigger */
  RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 



CREATE TRIGGER tD_Strain ON Strain FOR DELETE AS
/* ERwin Builtin Trigger */
/* DELETE trigger on Strain */
BEGIN
  DECLARE  @errno   int,
           @errmsg  varchar(255)
    /* ERwin Builtin Trigger */
    /* Strain  Worm on parent delete no action */
    /* ERWIN_RELATION:CHECKSUM="0000d6da", PARENT_OWNER="", PARENT_TABLE="Strain"
    CHILD_OWNER="", CHILD_TABLE="Worm"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_1", FK_COLUMNS="StrainKey" */
    IF EXISTS (
      SELECT * FROM deleted,Worm
      WHERE
        /*  %JoinFKPK(Worm,deleted," = "," AND") */
        Worm.StrainKey = deleted.StrainKey
    )
    BEGIN
      SELECT @errno  = 30001,
             @errmsg = 'Cannot delete Strain because Worm exists.'
      GOTO error
    END


    /* ERwin Builtin Trigger */
    RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 

CREATE TRIGGER tU_Strain ON Strain FOR UPDATE AS
/* ERwin Builtin Trigger */
/* UPDATE trigger on Strain */
BEGIN
  DECLARE  @numrows int,
           @nullcnt int,
           @validcnt int,
           @insStrainKey Key,
           @errno   int,
           @errmsg  varchar(255)

  SELECT @numrows = @@rowcount
  /* ERwin Builtin Trigger */
  /* Strain  Worm on parent update no action */
  /* ERWIN_RELATION:CHECKSUM="0000f5c9", PARENT_OWNER="", PARENT_TABLE="Strain"
    CHILD_OWNER="", CHILD_TABLE="Worm"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_1", FK_COLUMNS="StrainKey" */
  IF
    /* %ParentPK(" OR",UPDATE) */
    UPDATE(StrainKey)
  BEGIN
    IF EXISTS (
      SELECT * FROM deleted,Worm
      WHERE
        /*  %JoinFKPK(Worm,deleted," = "," AND") */
        Worm.StrainKey = deleted.StrainKey
    )
    BEGIN
      SELECT @errno  = 30005,
             @errmsg = 'Cannot update Strain because Worm exists.'
      GOTO error
    END
  END


  /* ERwin Builtin Trigger */
  RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 



CREATE TRIGGER tD_Type ON Type FOR DELETE AS
/* ERwin Builtin Trigger */
/* DELETE trigger on Type */
BEGIN
  DECLARE  @errno   int,
           @errmsg  varchar(255)
    /* ERwin Builtin Trigger */
    /* Type  WormFeature on parent delete no action */
    /* ERWIN_RELATION:CHECKSUM="0000efd0", PARENT_OWNER="", PARENT_TABLE="Type"
    CHILD_OWNER="", CHILD_TABLE="WormFeature"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_5", FK_COLUMNS="TypeKey" */
    IF EXISTS (
      SELECT * FROM deleted,WormFeature
      WHERE
        /*  %JoinFKPK(WormFeature,deleted," = "," AND") */
        WormFeature.TypeKey = deleted.TypeKey
    )
    BEGIN
      SELECT @errno  = 30001,
             @errmsg = 'Cannot delete Type because WormFeature exists.'
      GOTO error
    END


    /* ERwin Builtin Trigger */
    RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 

CREATE TRIGGER tU_Type ON Type FOR UPDATE AS
/* ERwin Builtin Trigger */
/* UPDATE trigger on Type */
BEGIN
  DECLARE  @numrows int,
           @nullcnt int,
           @validcnt int,
           @insTypeKey Key,
           @errno   int,
           @errmsg  varchar(255)

  SELECT @numrows = @@rowcount
  /* ERwin Builtin Trigger */
  /* Type  WormFeature on parent update no action */
  /* ERWIN_RELATION:CHECKSUM="00010ae8", PARENT_OWNER="", PARENT_TABLE="Type"
    CHILD_OWNER="", CHILD_TABLE="WormFeature"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_5", FK_COLUMNS="TypeKey" */
  IF
    /* %ParentPK(" OR",UPDATE) */
    UPDATE(TypeKey)
  BEGIN
    IF EXISTS (
      SELECT * FROM deleted,WormFeature
      WHERE
        /*  %JoinFKPK(WormFeature,deleted," = "," AND") */
        WormFeature.TypeKey = deleted.TypeKey
    )
    BEGIN
      SELECT @errno  = 30005,
             @errmsg = 'Cannot update Type because WormFeature exists.'
      GOTO error
    END
  END


  /* ERwin Builtin Trigger */
  RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 



CREATE TRIGGER tD_User ON User FOR DELETE AS
/* ERwin Builtin Trigger */
/* DELETE trigger on User */
BEGIN
  DECLARE  @errno   int,
           @errmsg  varchar(255)
    /* ERwin Builtin Trigger */
    /* Lab  User on child delete no action */
    /* ERWIN_RELATION:CHECKSUM="00010790", PARENT_OWNER="", PARENT_TABLE="Lab"
    CHILD_OWNER="", CHILD_TABLE="User"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_49", FK_COLUMNS="LabKey" */
    IF EXISTS (SELECT * FROM deleted,Lab
      WHERE
        /* %JoinFKPK(deleted,Lab," = "," AND") */
        deleted.LabKey = Lab.LabKey AND
        NOT EXISTS (
          SELECT * FROM User
          WHERE
            /* %JoinFKPK(User,Lab," = "," AND") */
            User.LabKey = Lab.LabKey
        )
    )
    BEGIN
      SELECT @errno  = 30010,
             @errmsg = 'Cannot delete last User because Lab exists.'
      GOTO error
    END


    /* ERwin Builtin Trigger */
    RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 

CREATE TRIGGER tU_User ON User FOR UPDATE AS
/* ERwin Builtin Trigger */
/* UPDATE trigger on User */
BEGIN
  DECLARE  @numrows int,
           @nullcnt int,
           @validcnt int,
           @insUserID Key,
           @errno   int,
           @errmsg  varchar(255)

  SELECT @numrows = @@rowcount
  /* ERwin Builtin Trigger */
  /* Lab  User on child update no action */
  /* ERWIN_RELATION:CHECKSUM="00012afb", PARENT_OWNER="", PARENT_TABLE="Lab"
    CHILD_OWNER="", CHILD_TABLE="User"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_49", FK_COLUMNS="LabKey" */
  IF
    /* %ChildFK(" OR",UPDATE) */
    UPDATE(LabKey)
  BEGIN
    SELECT @nullcnt = 0
    SELECT @validcnt = count(*)
      FROM inserted,Lab
        WHERE
          /* %JoinFKPK(inserted,Lab) */
          inserted.LabKey = Lab.LabKey
    /* %NotnullFK(inserted," IS NULL","select @nullcnt = count(*) from inserted where"," AND") */
    
    IF @validcnt + @nullcnt != @numrows
    BEGIN
      SELECT @errno  = 30007,
             @errmsg = 'Cannot update User because Lab does not exist.'
      GOTO error
    END
  END


  /* ERwin Builtin Trigger */
  RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 



CREATE TRIGGER tD_VideoAttributes ON VideoAttributes FOR DELETE AS
/* ERwin Builtin Trigger */
/* DELETE trigger on VideoAttributes */
BEGIN
  DECLARE  @errno   int,
           @errmsg  varchar(255)
    /* ERwin Builtin Trigger */
    /* VideoAttributes  PlateRawVideo on parent delete no action */
    /* ERWIN_RELATION:CHECKSUM="000105b3", PARENT_OWNER="", PARENT_TABLE="VideoAttributes"
    CHILD_OWNER="", CHILD_TABLE="PlateRawVideo"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_28", FK_COLUMNS="VideoMetadataKey" */
    IF EXISTS (
      SELECT * FROM deleted,PlateRawVideo
      WHERE
        /*  %JoinFKPK(PlateRawVideo,deleted," = "," AND") */
        PlateRawVideo.VideoMetadataKey = deleted.VideoMetadataKey
    )
    BEGIN
      SELECT @errno  = 30001,
             @errmsg = 'Cannot delete VideoAttributes because PlateRawVideo exists.'
      GOTO error
    END


    /* ERwin Builtin Trigger */
    RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 

CREATE TRIGGER tU_VideoAttributes ON VideoAttributes FOR UPDATE AS
/* ERwin Builtin Trigger */
/* UPDATE trigger on VideoAttributes */
BEGIN
  DECLARE  @numrows int,
           @nullcnt int,
           @validcnt int,
           @insVideoMetadataKey char(18),
           @errno   int,
           @errmsg  varchar(255)

  SELECT @numrows = @@rowcount
  /* ERwin Builtin Trigger */
  /* VideoAttributes  PlateRawVideo on parent update no action */
  /* ERWIN_RELATION:CHECKSUM="0001261d", PARENT_OWNER="", PARENT_TABLE="VideoAttributes"
    CHILD_OWNER="", CHILD_TABLE="PlateRawVideo"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_28", FK_COLUMNS="VideoMetadataKey" */
  IF
    /* %ParentPK(" OR",UPDATE) */
    UPDATE(VideoMetadataKey)
  BEGIN
    IF EXISTS (
      SELECT * FROM deleted,PlateRawVideo
      WHERE
        /*  %JoinFKPK(PlateRawVideo,deleted," = "," AND") */
        PlateRawVideo.VideoMetadataKey = deleted.VideoMetadataKey
    )
    BEGIN
      SELECT @errno  = 30005,
             @errmsg = 'Cannot update VideoAttributes because PlateRawVideo exists.'
      GOTO error
    END
  END


  /* ERwin Builtin Trigger */
  RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 



CREATE TRIGGER tD_WormFeature ON WormFeature FOR DELETE AS
/* ERwin Builtin Trigger */
/* DELETE trigger on WormFeature */
BEGIN
  DECLARE  @errno   int,
           @errmsg  varchar(255)
    /* ERwin Builtin Trigger */
    /* WormFeature  HistogramsPerWormWireframe on parent delete no action */
    /* ERWIN_RELATION:CHECKSUM="00082a10", PARENT_OWNER="", PARENT_TABLE="WormFeature"
    CHILD_OWNER="", CHILD_TABLE="HistogramsPerWormWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_27", FK_COLUMNS="WormFeatureKey" */
    IF EXISTS (
      SELECT * FROM deleted,HistogramsPerWormWireframe
      WHERE
        /*  %JoinFKPK(HistogramsPerWormWireframe,deleted," = "," AND") */
        HistogramsPerWormWireframe.WormFeatureKey = deleted.WormFeatureKey
    )
    BEGIN
      SELECT @errno  = 30001,
             @errmsg = 'Cannot delete WormFeature because HistogramsPerWormWireframe exists.'
      GOTO error
    END

    /* ERwin Builtin Trigger */
    /* WormFeature  FeaturesPerWormWireframe on parent delete no action */
    /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="WormFeature"
    CHILD_OWNER="", CHILD_TABLE="FeaturesPerWormWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_20", FK_COLUMNS="WormFeatureKey" */
    IF EXISTS (
      SELECT * FROM deleted,FeaturesPerWormWireframe
      WHERE
        /*  %JoinFKPK(FeaturesPerWormWireframe,deleted," = "," AND") */
        FeaturesPerWormWireframe.WormFeatureKey = deleted.WormFeatureKey
    )
    BEGIN
      SELECT @errno  = 30001,
             @errmsg = 'Cannot delete WormFeature because FeaturesPerWormWireframe exists.'
      GOTO error
    END

    /* ERwin Builtin Trigger */
    /* BodyPart  WormFeature on child delete no action */
    /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="BodyPart"
    CHILD_OWNER="", CHILD_TABLE="WormFeature"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_9", FK_COLUMNS="BodyPartKey" */
    IF EXISTS (SELECT * FROM deleted,BodyPart
      WHERE
        /* %JoinFKPK(deleted,BodyPart," = "," AND") */
        deleted.BodyPartKey = BodyPart.BodyPartKey AND
        NOT EXISTS (
          SELECT * FROM WormFeature
          WHERE
            /* %JoinFKPK(WormFeature,BodyPart," = "," AND") */
            WormFeature.BodyPartKey = BodyPart.BodyPartKey
        )
    )
    BEGIN
      SELECT @errno  = 30010,
             @errmsg = 'Cannot delete last WormFeature because BodyPart exists.'
      GOTO error
    END

    /* ERwin Builtin Trigger */
    /* Aspect  WormFeature on child delete no action */
    /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="Aspect"
    CHILD_OWNER="", CHILD_TABLE="WormFeature"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_8", FK_COLUMNS="AspectKey" */
    IF EXISTS (SELECT * FROM deleted,Aspect
      WHERE
        /* %JoinFKPK(deleted,Aspect," = "," AND") */
        deleted.AspectKey = Aspect.AspectKey AND
        NOT EXISTS (
          SELECT * FROM WormFeature
          WHERE
            /* %JoinFKPK(WormFeature,Aspect," = "," AND") */
            WormFeature.AspectKey = Aspect.AspectKey
        )
    )
    BEGIN
      SELECT @errno  = 30010,
             @errmsg = 'Cannot delete last WormFeature because Aspect exists.'
      GOTO error
    END

    /* ERwin Builtin Trigger */
    /* Direction  WormFeature on child delete no action */
    /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="Direction"
    CHILD_OWNER="", CHILD_TABLE="WormFeature"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_7", FK_COLUMNS="DirectionKey" */
    IF EXISTS (SELECT * FROM deleted,Direction
      WHERE
        /* %JoinFKPK(deleted,Direction," = "," AND") */
        deleted.DirectionKey = Direction.DirectionKey AND
        NOT EXISTS (
          SELECT * FROM WormFeature
          WHERE
            /* %JoinFKPK(WormFeature,Direction," = "," AND") */
            WormFeature.DirectionKey = Direction.DirectionKey
        )
    )
    BEGIN
      SELECT @errno  = 30010,
             @errmsg = 'Cannot delete last WormFeature because Direction exists.'
      GOTO error
    END

    /* ERwin Builtin Trigger */
    /* Category  WormFeature on child delete no action */
    /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="Category"
    CHILD_OWNER="", CHILD_TABLE="WormFeature"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_6", FK_COLUMNS="CategoryKey" */
    IF EXISTS (SELECT * FROM deleted,Category
      WHERE
        /* %JoinFKPK(deleted,Category," = "," AND") */
        deleted.CategoryKey = Category.CategoryKey AND
        NOT EXISTS (
          SELECT * FROM WormFeature
          WHERE
            /* %JoinFKPK(WormFeature,Category," = "," AND") */
            WormFeature.CategoryKey = Category.CategoryKey
        )
    )
    BEGIN
      SELECT @errno  = 30010,
             @errmsg = 'Cannot delete last WormFeature because Category exists.'
      GOTO error
    END

    /* ERwin Builtin Trigger */
    /* Type  WormFeature on child delete no action */
    /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="Type"
    CHILD_OWNER="", CHILD_TABLE="WormFeature"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_5", FK_COLUMNS="TypeKey" */
    IF EXISTS (SELECT * FROM deleted,Type
      WHERE
        /* %JoinFKPK(deleted,Type," = "," AND") */
        deleted.TypeKey = Type.TypeKey AND
        NOT EXISTS (
          SELECT * FROM WormFeature
          WHERE
            /* %JoinFKPK(WormFeature,Type," = "," AND") */
            WormFeature.TypeKey = Type.TypeKey
        )
    )
    BEGIN
      SELECT @errno  = 30010,
             @errmsg = 'Cannot delete last WormFeature because Type exists.'
      GOTO error
    END


    /* ERwin Builtin Trigger */
    RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 

CREATE TRIGGER tU_WormFeature ON WormFeature FOR UPDATE AS
/* ERwin Builtin Trigger */
/* UPDATE trigger on WormFeature */
BEGIN
  DECLARE  @numrows int,
           @nullcnt int,
           @validcnt int,
           @insWormFeatureKey Key,
           @errno   int,
           @errmsg  varchar(255)

  SELECT @numrows = @@rowcount
  /* ERwin Builtin Trigger */
  /* WormFeature  HistogramsPerWormWireframe on parent update no action */
  /* ERWIN_RELATION:CHECKSUM="0008f158", PARENT_OWNER="", PARENT_TABLE="WormFeature"
    CHILD_OWNER="", CHILD_TABLE="HistogramsPerWormWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_27", FK_COLUMNS="WormFeatureKey" */
  IF
    /* %ParentPK(" OR",UPDATE) */
    UPDATE(WormFeatureKey)
  BEGIN
    IF EXISTS (
      SELECT * FROM deleted,HistogramsPerWormWireframe
      WHERE
        /*  %JoinFKPK(HistogramsPerWormWireframe,deleted," = "," AND") */
        HistogramsPerWormWireframe.WormFeatureKey = deleted.WormFeatureKey
    )
    BEGIN
      SELECT @errno  = 30005,
             @errmsg = 'Cannot update WormFeature because HistogramsPerWormWireframe exists.'
      GOTO error
    END
  END

  /* ERwin Builtin Trigger */
  /* WormFeature  FeaturesPerWormWireframe on parent update no action */
  /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="WormFeature"
    CHILD_OWNER="", CHILD_TABLE="FeaturesPerWormWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_20", FK_COLUMNS="WormFeatureKey" */
  IF
    /* %ParentPK(" OR",UPDATE) */
    UPDATE(WormFeatureKey)
  BEGIN
    IF EXISTS (
      SELECT * FROM deleted,FeaturesPerWormWireframe
      WHERE
        /*  %JoinFKPK(FeaturesPerWormWireframe,deleted," = "," AND") */
        FeaturesPerWormWireframe.WormFeatureKey = deleted.WormFeatureKey
    )
    BEGIN
      SELECT @errno  = 30005,
             @errmsg = 'Cannot update WormFeature because FeaturesPerWormWireframe exists.'
      GOTO error
    END
  END

  /* ERwin Builtin Trigger */
  /* BodyPart  WormFeature on child update no action */
  /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="BodyPart"
    CHILD_OWNER="", CHILD_TABLE="WormFeature"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_9", FK_COLUMNS="BodyPartKey" */
  IF
    /* %ChildFK(" OR",UPDATE) */
    UPDATE(BodyPartKey)
  BEGIN
    SELECT @nullcnt = 0
    SELECT @validcnt = count(*)
      FROM inserted,BodyPart
        WHERE
          /* %JoinFKPK(inserted,BodyPart) */
          inserted.BodyPartKey = BodyPart.BodyPartKey
    /* %NotnullFK(inserted," IS NULL","select @nullcnt = count(*) from inserted where"," AND") */
    
    IF @validcnt + @nullcnt != @numrows
    BEGIN
      SELECT @errno  = 30007,
             @errmsg = 'Cannot update WormFeature because BodyPart does not exist.'
      GOTO error
    END
  END

  /* ERwin Builtin Trigger */
  /* Aspect  WormFeature on child update no action */
  /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="Aspect"
    CHILD_OWNER="", CHILD_TABLE="WormFeature"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_8", FK_COLUMNS="AspectKey" */
  IF
    /* %ChildFK(" OR",UPDATE) */
    UPDATE(AspectKey)
  BEGIN
    SELECT @nullcnt = 0
    SELECT @validcnt = count(*)
      FROM inserted,Aspect
        WHERE
          /* %JoinFKPK(inserted,Aspect) */
          inserted.AspectKey = Aspect.AspectKey
    /* %NotnullFK(inserted," IS NULL","select @nullcnt = count(*) from inserted where"," AND") */
    
    IF @validcnt + @nullcnt != @numrows
    BEGIN
      SELECT @errno  = 30007,
             @errmsg = 'Cannot update WormFeature because Aspect does not exist.'
      GOTO error
    END
  END

  /* ERwin Builtin Trigger */
  /* Direction  WormFeature on child update no action */
  /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="Direction"
    CHILD_OWNER="", CHILD_TABLE="WormFeature"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_7", FK_COLUMNS="DirectionKey" */
  IF
    /* %ChildFK(" OR",UPDATE) */
    UPDATE(DirectionKey)
  BEGIN
    SELECT @nullcnt = 0
    SELECT @validcnt = count(*)
      FROM inserted,Direction
        WHERE
          /* %JoinFKPK(inserted,Direction) */
          inserted.DirectionKey = Direction.DirectionKey
    /* %NotnullFK(inserted," IS NULL","select @nullcnt = count(*) from inserted where"," AND") */
    
    IF @validcnt + @nullcnt != @numrows
    BEGIN
      SELECT @errno  = 30007,
             @errmsg = 'Cannot update WormFeature because Direction does not exist.'
      GOTO error
    END
  END

  /* ERwin Builtin Trigger */
  /* Category  WormFeature on child update no action */
  /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="Category"
    CHILD_OWNER="", CHILD_TABLE="WormFeature"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_6", FK_COLUMNS="CategoryKey" */
  IF
    /* %ChildFK(" OR",UPDATE) */
    UPDATE(CategoryKey)
  BEGIN
    SELECT @nullcnt = 0
    SELECT @validcnt = count(*)
      FROM inserted,Category
        WHERE
          /* %JoinFKPK(inserted,Category) */
          inserted.CategoryKey = Category.CategoryKey
    /* %NotnullFK(inserted," IS NULL","select @nullcnt = count(*) from inserted where"," AND") */
    
    IF @validcnt + @nullcnt != @numrows
    BEGIN
      SELECT @errno  = 30007,
             @errmsg = 'Cannot update WormFeature because Category does not exist.'
      GOTO error
    END
  END

  /* ERwin Builtin Trigger */
  /* Type  WormFeature on child update no action */
  /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="Type"
    CHILD_OWNER="", CHILD_TABLE="WormFeature"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_5", FK_COLUMNS="TypeKey" */
  IF
    /* %ChildFK(" OR",UPDATE) */
    UPDATE(TypeKey)
  BEGIN
    SELECT @nullcnt = 0
    SELECT @validcnt = count(*)
      FROM inserted,Type
        WHERE
          /* %JoinFKPK(inserted,Type) */
          inserted.TypeKey = Type.TypeKey
    /* %NotnullFK(inserted," IS NULL","select @nullcnt = count(*) from inserted where"," AND") */
    
    IF @validcnt + @nullcnt != @numrows
    BEGIN
      SELECT @errno  = 30007,
             @errmsg = 'Cannot update WormFeature because Type does not exist.'
      GOTO error
    END
  END


  /* ERwin Builtin Trigger */
  RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 



CREATE TRIGGER tD_Worm ON Worm FOR DELETE AS
/* ERwin Builtin Trigger */
/* DELETE trigger on Worm */
BEGIN
  DECLARE  @errno   int,
           @errmsg  varchar(255)
    /* ERwin Builtin Trigger */
    /* Worm  WormList on parent delete no action */
    /* ERWIN_RELATION:CHECKSUM="00020ae8", PARENT_OWNER="", PARENT_TABLE="Worm"
    CHILD_OWNER="", CHILD_TABLE="WormList"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_45", FK_COLUMNS="WormKey" */
    IF EXISTS (
      SELECT * FROM deleted,WormList
      WHERE
        /*  %JoinFKPK(WormList,deleted," = "," AND") */
        WormList.WormKey = deleted.WormKey
    )
    BEGIN
      SELECT @errno  = 30001,
             @errmsg = 'Cannot delete Worm because WormList exists.'
      GOTO error
    END

    /* ERwin Builtin Trigger */
    /* Strain  Worm on child delete no action */
    /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="Strain"
    CHILD_OWNER="", CHILD_TABLE="Worm"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_1", FK_COLUMNS="StrainKey" */
    IF EXISTS (SELECT * FROM deleted,Strain
      WHERE
        /* %JoinFKPK(deleted,Strain," = "," AND") */
        deleted.StrainKey = Strain.StrainKey AND
        NOT EXISTS (
          SELECT * FROM Worm
          WHERE
            /* %JoinFKPK(Worm,Strain," = "," AND") */
            Worm.StrainKey = Strain.StrainKey
        )
    )
    BEGIN
      SELECT @errno  = 30010,
             @errmsg = 'Cannot delete last Worm because Strain exists.'
      GOTO error
    END


    /* ERwin Builtin Trigger */
    RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 

CREATE TRIGGER tU_Worm ON Worm FOR UPDATE AS
/* ERwin Builtin Trigger */
/* UPDATE trigger on Worm */
BEGIN
  DECLARE  @numrows int,
           @nullcnt int,
           @validcnt int,
           @insWormKey Key,
           @errno   int,
           @errmsg  varchar(255)

  SELECT @numrows = @@rowcount
  /* ERwin Builtin Trigger */
  /* Worm  WormList on parent update no action */
  /* ERWIN_RELATION:CHECKSUM="00023fe0", PARENT_OWNER="", PARENT_TABLE="Worm"
    CHILD_OWNER="", CHILD_TABLE="WormList"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_45", FK_COLUMNS="WormKey" */
  IF
    /* %ParentPK(" OR",UPDATE) */
    UPDATE(WormKey)
  BEGIN
    IF EXISTS (
      SELECT * FROM deleted,WormList
      WHERE
        /*  %JoinFKPK(WormList,deleted," = "," AND") */
        WormList.WormKey = deleted.WormKey
    )
    BEGIN
      SELECT @errno  = 30005,
             @errmsg = 'Cannot update Worm because WormList exists.'
      GOTO error
    END
  END

  /* ERwin Builtin Trigger */
  /* Strain  Worm on child update no action */
  /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="Strain"
    CHILD_OWNER="", CHILD_TABLE="Worm"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_1", FK_COLUMNS="StrainKey" */
  IF
    /* %ChildFK(" OR",UPDATE) */
    UPDATE(StrainKey)
  BEGIN
    SELECT @nullcnt = 0
    SELECT @validcnt = count(*)
      FROM inserted,Strain
        WHERE
          /* %JoinFKPK(inserted,Strain) */
          inserted.StrainKey = Strain.StrainKey
    /* %NotnullFK(inserted," IS NULL","select @nullcnt = count(*) from inserted where"," AND") */
    
    IF @validcnt + @nullcnt != @numrows
    BEGIN
      SELECT @errno  = 30007,
             @errmsg = 'Cannot update Worm because Strain does not exist.'
      GOTO error
    END
  END


  /* ERwin Builtin Trigger */
  RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 



CREATE TRIGGER tD_WormList ON WormList FOR DELETE AS
/* ERwin Builtin Trigger */
/* DELETE trigger on WormList */
BEGIN
  DECLARE  @errno   int,
           @errmsg  varchar(255)
    /* ERwin Builtin Trigger */
    /* WormList  WormInteraction on parent delete no action */
    /* ERWIN_RELATION:CHECKSUM="00030489", PARENT_OWNER="", PARENT_TABLE="WormList"
    CHILD_OWNER="", CHILD_TABLE="WormInteraction"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_47", FK_COLUMNS="WormListKey" */
    IF EXISTS (
      SELECT * FROM deleted,WormInteraction
      WHERE
        /*  %JoinFKPK(WormInteraction,deleted," = "," AND") */
        WormInteraction.WormListKey = deleted.WormListKey
    )
    BEGIN
      SELECT @errno  = 30001,
             @errmsg = 'Cannot delete WormList because WormInteraction exists.'
      GOTO error
    END

    /* ERwin Builtin Trigger */
    /* WormList  Plate on parent delete no action */
    /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="WormList"
    CHILD_OWNER="", CHILD_TABLE="Plate"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_46", FK_COLUMNS="WormListKey" */
    IF EXISTS (
      SELECT * FROM deleted,Plate
      WHERE
        /*  %JoinFKPK(Plate,deleted," = "," AND") */
        Plate.WormListKey = deleted.WormListKey
    )
    BEGIN
      SELECT @errno  = 30001,
             @errmsg = 'Cannot delete WormList because Plate exists.'
      GOTO error
    END

    /* ERwin Builtin Trigger */
    /* Worm  WormList on child delete no action */
    /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="Worm"
    CHILD_OWNER="", CHILD_TABLE="WormList"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_45", FK_COLUMNS="WormKey" */
    IF EXISTS (SELECT * FROM deleted,Worm
      WHERE
        /* %JoinFKPK(deleted,Worm," = "," AND") */
        deleted.WormKey = Worm.WormKey AND
        NOT EXISTS (
          SELECT * FROM WormList
          WHERE
            /* %JoinFKPK(WormList,Worm," = "," AND") */
            WormList.WormKey = Worm.WormKey
        )
    )
    BEGIN
      SELECT @errno  = 30010,
             @errmsg = 'Cannot delete last WormList because Worm exists.'
      GOTO error
    END


    /* ERwin Builtin Trigger */
    RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 

CREATE TRIGGER tU_WormList ON WormList FOR UPDATE AS
/* ERwin Builtin Trigger */
/* UPDATE trigger on WormList */
BEGIN
  DECLARE  @numrows int,
           @nullcnt int,
           @validcnt int,
           @insWormListKey Key,
           @errno   int,
           @errmsg  varchar(255)

  SELECT @numrows = @@rowcount
  /* ERwin Builtin Trigger */
  /* WormList  WormInteraction on parent update no action */
  /* ERWIN_RELATION:CHECKSUM="00035abf", PARENT_OWNER="", PARENT_TABLE="WormList"
    CHILD_OWNER="", CHILD_TABLE="WormInteraction"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_47", FK_COLUMNS="WormListKey" */
  IF
    /* %ParentPK(" OR",UPDATE) */
    UPDATE(WormListKey)
  BEGIN
    IF EXISTS (
      SELECT * FROM deleted,WormInteraction
      WHERE
        /*  %JoinFKPK(WormInteraction,deleted," = "," AND") */
        WormInteraction.WormListKey = deleted.WormListKey
    )
    BEGIN
      SELECT @errno  = 30005,
             @errmsg = 'Cannot update WormList because WormInteraction exists.'
      GOTO error
    END
  END

  /* ERwin Builtin Trigger */
  /* WormList  Plate on parent update no action */
  /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="WormList"
    CHILD_OWNER="", CHILD_TABLE="Plate"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_46", FK_COLUMNS="WormListKey" */
  IF
    /* %ParentPK(" OR",UPDATE) */
    UPDATE(WormListKey)
  BEGIN
    IF EXISTS (
      SELECT * FROM deleted,Plate
      WHERE
        /*  %JoinFKPK(Plate,deleted," = "," AND") */
        Plate.WormListKey = deleted.WormListKey
    )
    BEGIN
      SELECT @errno  = 30005,
             @errmsg = 'Cannot update WormList because Plate exists.'
      GOTO error
    END
  END

  /* ERwin Builtin Trigger */
  /* Worm  WormList on child update no action */
  /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="Worm"
    CHILD_OWNER="", CHILD_TABLE="WormList"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_45", FK_COLUMNS="WormKey" */
  IF
    /* %ChildFK(" OR",UPDATE) */
    UPDATE(WormKey)
  BEGIN
    SELECT @nullcnt = 0
    SELECT @validcnt = count(*)
      FROM inserted,Worm
        WHERE
          /* %JoinFKPK(inserted,Worm) */
          inserted.WormKey = Worm.WormKey
    /* %NotnullFK(inserted," IS NULL","select @nullcnt = count(*) from inserted where"," AND") */
    
    IF @validcnt + @nullcnt != @numrows
    BEGIN
      SELECT @errno  = 30007,
             @errmsg = 'Cannot update WormList because Worm does not exist.'
      GOTO error
    END
  END


  /* ERwin Builtin Trigger */
  RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 



CREATE TRIGGER tD_PlateWireframeVideo ON PlateWireframeVideo FOR DELETE AS
/* ERwin Builtin Trigger */
/* DELETE trigger on PlateWireframeVideo */
BEGIN
  DECLARE  @errno   int,
           @errmsg  varchar(255)
    /* ERwin Builtin Trigger */
    /* PlateWireframeVideo  HistogramsPerPlateWireframe on parent delete no action */
    /* ERWIN_RELATION:CHECKSUM="0007b749", PARENT_OWNER="", PARENT_TABLE="PlateWireframeVideo"
    CHILD_OWNER="", CHILD_TABLE="HistogramsPerPlateWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_41", FK_COLUMNS="PlateWireframeVideoKey" */
    IF EXISTS (
      SELECT * FROM deleted,HistogramsPerPlateWireframe
      WHERE
        /*  %JoinFKPK(HistogramsPerPlateWireframe,deleted," = "," AND") */
        HistogramsPerPlateWireframe.PlateWireframeVideoKey = deleted.PlateWireframeVideoKey
    )
    BEGIN
      SELECT @errno  = 30001,
             @errmsg = 'Cannot delete PlateWireframeVideo because HistogramsPerPlateWireframe exists.'
      GOTO error
    END

    /* ERwin Builtin Trigger */
    /* PlateWireframeVideo  WormWireframeVideo on parent delete no action */
    /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="PlateWireframeVideo"
    CHILD_OWNER="", CHILD_TABLE="WormWireframeVideo"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_40", FK_COLUMNS="PlateWireframeVideoKey" */
    IF EXISTS (
      SELECT * FROM deleted,WormWireframeVideo
      WHERE
        /*  %JoinFKPK(WormWireframeVideo,deleted," = "," AND") */
        WormWireframeVideo.PlateWireframeVideoKey = deleted.PlateWireframeVideoKey
    )
    BEGIN
      SELECT @errno  = 30001,
             @errmsg = 'Cannot delete PlateWireframeVideo because WormWireframeVideo exists.'
      GOTO error
    END

    /* ERwin Builtin Trigger */
    /* PlateWireframeVideo  WormInteraction on parent delete no action */
    /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="PlateWireframeVideo"
    CHILD_OWNER="", CHILD_TABLE="WormInteraction"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_39", FK_COLUMNS="PlateWireframeVideoKey" */
    IF EXISTS (
      SELECT * FROM deleted,WormInteraction
      WHERE
        /*  %JoinFKPK(WormInteraction,deleted," = "," AND") */
        WormInteraction.PlateWireframeVideoKey = deleted.PlateWireframeVideoKey
    )
    BEGIN
      SELECT @errno  = 30001,
             @errmsg = 'Cannot delete PlateWireframeVideo because WormInteraction exists.'
      GOTO error
    END

    /* ERwin Builtin Trigger */
    /* PlateWireframeVideo  FeaturesPerPlateWireframe on parent delete no action */
    /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="PlateWireframeVideo"
    CHILD_OWNER="", CHILD_TABLE="FeaturesPerPlateWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_36", FK_COLUMNS="PlateWireframeVideoKey" */
    IF EXISTS (
      SELECT * FROM deleted,FeaturesPerPlateWireframe
      WHERE
        /*  %JoinFKPK(FeaturesPerPlateWireframe,deleted," = "," AND") */
        FeaturesPerPlateWireframe.PlateWireframeVideoKey = deleted.PlateWireframeVideoKey
    )
    BEGIN
      SELECT @errno  = 30001,
             @errmsg = 'Cannot delete PlateWireframeVideo because FeaturesPerPlateWireframe exists.'
      GOTO error
    END

    /* ERwin Builtin Trigger */
    /* ComputerVisionAlgorithm  PlateWireframeVideo on child delete no action */
    /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="ComputerVisionAlgorithm"
    CHILD_OWNER="", CHILD_TABLE="PlateWireframeVideo"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_15", FK_COLUMNS="CVAlgorithmKey" */
    IF EXISTS (SELECT * FROM deleted,ComputerVisionAlgorithm
      WHERE
        /* %JoinFKPK(deleted,ComputerVisionAlgorithm," = "," AND") */
        deleted.CVAlgorithmKey = ComputerVisionAlgorithm.CVAlgorithmKey AND
        NOT EXISTS (
          SELECT * FROM PlateWireframeVideo
          WHERE
            /* %JoinFKPK(PlateWireframeVideo,ComputerVisionAlgorithm," = "," AND") */
            PlateWireframeVideo.CVAlgorithmKey = ComputerVisionAlgorithm.CVAlgorithmKey
        )
    )
    BEGIN
      SELECT @errno  = 30010,
             @errmsg = 'Cannot delete last PlateWireframeVideo because ComputerVisionAlgorithm exists.'
      GOTO error
    END

    /* ERwin Builtin Trigger */
    /* PlateRawVideo  PlateWireframeVideo on child delete no action */
    /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="PlateRawVideo"
    CHILD_OWNER="", CHILD_TABLE="PlateWireframeVideo"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_14", FK_COLUMNS="PlateRawVideoKey" */
    IF EXISTS (SELECT * FROM deleted,PlateRawVideo
      WHERE
        /* %JoinFKPK(deleted,PlateRawVideo," = "," AND") */
        deleted.PlateRawVideoKey = PlateRawVideo.PlateRawVideoKey AND
        NOT EXISTS (
          SELECT * FROM PlateWireframeVideo
          WHERE
            /* %JoinFKPK(PlateWireframeVideo,PlateRawVideo," = "," AND") */
            PlateWireframeVideo.PlateRawVideoKey = PlateRawVideo.PlateRawVideoKey
        )
    )
    BEGIN
      SELECT @errno  = 30010,
             @errmsg = 'Cannot delete last PlateWireframeVideo because PlateRawVideo exists.'
      GOTO error
    END


    /* ERwin Builtin Trigger */
    RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 

CREATE TRIGGER tU_PlateWireframeVideo ON PlateWireframeVideo FOR UPDATE AS
/* ERwin Builtin Trigger */
/* UPDATE trigger on PlateWireframeVideo */
BEGIN
  DECLARE  @numrows int,
           @nullcnt int,
           @validcnt int,
           @insPlateWireframeVideoKey Key,
           @errno   int,
           @errmsg  varchar(255)

  SELECT @numrows = @@rowcount
  /* ERwin Builtin Trigger */
  /* PlateWireframeVideo  HistogramsPerPlateWireframe on parent update no action */
  /* ERWIN_RELATION:CHECKSUM="000842cb", PARENT_OWNER="", PARENT_TABLE="PlateWireframeVideo"
    CHILD_OWNER="", CHILD_TABLE="HistogramsPerPlateWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_41", FK_COLUMNS="PlateWireframeVideoKey" */
  IF
    /* %ParentPK(" OR",UPDATE) */
    UPDATE(PlateWireframeVideoKey)
  BEGIN
    IF EXISTS (
      SELECT * FROM deleted,HistogramsPerPlateWireframe
      WHERE
        /*  %JoinFKPK(HistogramsPerPlateWireframe,deleted," = "," AND") */
        HistogramsPerPlateWireframe.PlateWireframeVideoKey = deleted.PlateWireframeVideoKey
    )
    BEGIN
      SELECT @errno  = 30005,
             @errmsg = 'Cannot update PlateWireframeVideo because HistogramsPerPlateWireframe exists.'
      GOTO error
    END
  END

  /* ERwin Builtin Trigger */
  /* PlateWireframeVideo  WormWireframeVideo on parent update no action */
  /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="PlateWireframeVideo"
    CHILD_OWNER="", CHILD_TABLE="WormWireframeVideo"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_40", FK_COLUMNS="PlateWireframeVideoKey" */
  IF
    /* %ParentPK(" OR",UPDATE) */
    UPDATE(PlateWireframeVideoKey)
  BEGIN
    IF EXISTS (
      SELECT * FROM deleted,WormWireframeVideo
      WHERE
        /*  %JoinFKPK(WormWireframeVideo,deleted," = "," AND") */
        WormWireframeVideo.PlateWireframeVideoKey = deleted.PlateWireframeVideoKey
    )
    BEGIN
      SELECT @errno  = 30005,
             @errmsg = 'Cannot update PlateWireframeVideo because WormWireframeVideo exists.'
      GOTO error
    END
  END

  /* ERwin Builtin Trigger */
  /* PlateWireframeVideo  WormInteraction on parent update no action */
  /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="PlateWireframeVideo"
    CHILD_OWNER="", CHILD_TABLE="WormInteraction"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_39", FK_COLUMNS="PlateWireframeVideoKey" */
  IF
    /* %ParentPK(" OR",UPDATE) */
    UPDATE(PlateWireframeVideoKey)
  BEGIN
    IF EXISTS (
      SELECT * FROM deleted,WormInteraction
      WHERE
        /*  %JoinFKPK(WormInteraction,deleted," = "," AND") */
        WormInteraction.PlateWireframeVideoKey = deleted.PlateWireframeVideoKey
    )
    BEGIN
      SELECT @errno  = 30005,
             @errmsg = 'Cannot update PlateWireframeVideo because WormInteraction exists.'
      GOTO error
    END
  END

  /* ERwin Builtin Trigger */
  /* PlateWireframeVideo  FeaturesPerPlateWireframe on parent update no action */
  /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="PlateWireframeVideo"
    CHILD_OWNER="", CHILD_TABLE="FeaturesPerPlateWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_36", FK_COLUMNS="PlateWireframeVideoKey" */
  IF
    /* %ParentPK(" OR",UPDATE) */
    UPDATE(PlateWireframeVideoKey)
  BEGIN
    IF EXISTS (
      SELECT * FROM deleted,FeaturesPerPlateWireframe
      WHERE
        /*  %JoinFKPK(FeaturesPerPlateWireframe,deleted," = "," AND") */
        FeaturesPerPlateWireframe.PlateWireframeVideoKey = deleted.PlateWireframeVideoKey
    )
    BEGIN
      SELECT @errno  = 30005,
             @errmsg = 'Cannot update PlateWireframeVideo because FeaturesPerPlateWireframe exists.'
      GOTO error
    END
  END

  /* ERwin Builtin Trigger */
  /* ComputerVisionAlgorithm  PlateWireframeVideo on child update no action */
  /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="ComputerVisionAlgorithm"
    CHILD_OWNER="", CHILD_TABLE="PlateWireframeVideo"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_15", FK_COLUMNS="CVAlgorithmKey" */
  IF
    /* %ChildFK(" OR",UPDATE) */
    UPDATE(CVAlgorithmKey)
  BEGIN
    SELECT @nullcnt = 0
    SELECT @validcnt = count(*)
      FROM inserted,ComputerVisionAlgorithm
        WHERE
          /* %JoinFKPK(inserted,ComputerVisionAlgorithm) */
          inserted.CVAlgorithmKey = ComputerVisionAlgorithm.CVAlgorithmKey
    /* %NotnullFK(inserted," IS NULL","select @nullcnt = count(*) from inserted where"," AND") */
    
    IF @validcnt + @nullcnt != @numrows
    BEGIN
      SELECT @errno  = 30007,
             @errmsg = 'Cannot update PlateWireframeVideo because ComputerVisionAlgorithm does not exist.'
      GOTO error
    END
  END

  /* ERwin Builtin Trigger */
  /* PlateRawVideo  PlateWireframeVideo on child update no action */
  /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="PlateRawVideo"
    CHILD_OWNER="", CHILD_TABLE="PlateWireframeVideo"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_14", FK_COLUMNS="PlateRawVideoKey" */
  IF
    /* %ChildFK(" OR",UPDATE) */
    UPDATE(PlateRawVideoKey)
  BEGIN
    SELECT @nullcnt = 0
    SELECT @validcnt = count(*)
      FROM inserted,PlateRawVideo
        WHERE
          /* %JoinFKPK(inserted,PlateRawVideo) */
          inserted.PlateRawVideoKey = PlateRawVideo.PlateRawVideoKey
    /* %NotnullFK(inserted," IS NULL","select @nullcnt = count(*) from inserted where"," AND") */
    
    IF @validcnt + @nullcnt != @numrows
    BEGIN
      SELECT @errno  = 30007,
             @errmsg = 'Cannot update PlateWireframeVideo because PlateRawVideo does not exist.'
      GOTO error
    END
  END


  /* ERwin Builtin Trigger */
  RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 



CREATE TRIGGER tD_WormInteraction ON WormInteraction FOR DELETE AS
/* ERwin Builtin Trigger */
/* DELETE trigger on WormInteraction */
BEGIN
  DECLARE  @errno   int,
           @errmsg  varchar(255)
    /* ERwin Builtin Trigger */
    /* WormList  WormInteraction on child delete no action */
    /* ERWIN_RELATION:CHECKSUM="0002bfee", PARENT_OWNER="", PARENT_TABLE="WormList"
    CHILD_OWNER="", CHILD_TABLE="WormInteraction"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_47", FK_COLUMNS="WormListKey" */
    IF EXISTS (SELECT * FROM deleted,WormList
      WHERE
        /* %JoinFKPK(deleted,WormList," = "," AND") */
        deleted.WormListKey = WormList.WormListKey AND
        NOT EXISTS (
          SELECT * FROM WormInteraction
          WHERE
            /* %JoinFKPK(WormInteraction,WormList," = "," AND") */
            WormInteraction.WormListKey = WormList.WormListKey
        )
    )
    BEGIN
      SELECT @errno  = 30010,
             @errmsg = 'Cannot delete last WormInteraction because WormList exists.'
      GOTO error
    END

    /* ERwin Builtin Trigger */
    /* PlateWireframeVideo  WormInteraction on child delete no action */
    /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="PlateWireframeVideo"
    CHILD_OWNER="", CHILD_TABLE="WormInteraction"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_39", FK_COLUMNS="PlateWireframeVideoKey" */
    IF EXISTS (SELECT * FROM deleted,PlateWireframeVideo
      WHERE
        /* %JoinFKPK(deleted,PlateWireframeVideo," = "," AND") */
        deleted.PlateWireframeVideoKey = PlateWireframeVideo.PlateWireframeVideoKey AND
        NOT EXISTS (
          SELECT * FROM WormInteraction
          WHERE
            /* %JoinFKPK(WormInteraction,PlateWireframeVideo," = "," AND") */
            WormInteraction.PlateWireframeVideoKey = PlateWireframeVideo.PlateWireframeVideoKey
        )
    )
    BEGIN
      SELECT @errno  = 30010,
             @errmsg = 'Cannot delete last WormInteraction because PlateWireframeVideo exists.'
      GOTO error
    END


    /* ERwin Builtin Trigger */
    RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 

CREATE TRIGGER tU_WormInteraction ON WormInteraction FOR UPDATE AS
/* ERwin Builtin Trigger */
/* UPDATE trigger on WormInteraction */
BEGIN
  DECLARE  @numrows int,
           @nullcnt int,
           @validcnt int,
           @insWormInteractionKey char(18),
           @errno   int,
           @errmsg  varchar(255)

  SELECT @numrows = @@rowcount
  /* ERwin Builtin Trigger */
  /* WormList  WormInteraction on child update no action */
  /* ERWIN_RELATION:CHECKSUM="0002ed86", PARENT_OWNER="", PARENT_TABLE="WormList"
    CHILD_OWNER="", CHILD_TABLE="WormInteraction"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_47", FK_COLUMNS="WormListKey" */
  IF
    /* %ChildFK(" OR",UPDATE) */
    UPDATE(WormListKey)
  BEGIN
    SELECT @nullcnt = 0
    SELECT @validcnt = count(*)
      FROM inserted,WormList
        WHERE
          /* %JoinFKPK(inserted,WormList) */
          inserted.WormListKey = WormList.WormListKey
    /* %NotnullFK(inserted," IS NULL","select @nullcnt = count(*) from inserted where"," AND") */
    
    IF @validcnt + @nullcnt != @numrows
    BEGIN
      SELECT @errno  = 30007,
             @errmsg = 'Cannot update WormInteraction because WormList does not exist.'
      GOTO error
    END
  END

  /* ERwin Builtin Trigger */
  /* PlateWireframeVideo  WormInteraction on child update no action */
  /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="PlateWireframeVideo"
    CHILD_OWNER="", CHILD_TABLE="WormInteraction"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_39", FK_COLUMNS="PlateWireframeVideoKey" */
  IF
    /* %ChildFK(" OR",UPDATE) */
    UPDATE(PlateWireframeVideoKey)
  BEGIN
    SELECT @nullcnt = 0
    SELECT @validcnt = count(*)
      FROM inserted,PlateWireframeVideo
        WHERE
          /* %JoinFKPK(inserted,PlateWireframeVideo) */
          inserted.PlateWireframeVideoKey = PlateWireframeVideo.PlateWireframeVideoKey
    /* %NotnullFK(inserted," IS NULL","select @nullcnt = count(*) from inserted where"," AND") */
    
    IF @validcnt + @nullcnt != @numrows
    BEGIN
      SELECT @errno  = 30007,
             @errmsg = 'Cannot update WormInteraction because PlateWireframeVideo does not exist.'
      GOTO error
    END
  END


  /* ERwin Builtin Trigger */
  RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 



CREATE TRIGGER tD_FeaturesPerPlateWireframe ON FeaturesPerPlateWireframe FOR DELETE AS
/* ERwin Builtin Trigger */
/* DELETE trigger on FeaturesPerPlateWireframe */
BEGIN
  DECLARE  @errno   int,
           @errmsg  varchar(255)
    /* ERwin Builtin Trigger */
    /* PlateWireframeVideo  FeaturesPerPlateWireframe on child delete no action */
    /* ERWIN_RELATION:CHECKSUM="0003204a", PARENT_OWNER="", PARENT_TABLE="PlateWireframeVideo"
    CHILD_OWNER="", CHILD_TABLE="FeaturesPerPlateWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_36", FK_COLUMNS="PlateWireframeVideoKey" */
    IF EXISTS (SELECT * FROM deleted,PlateWireframeVideo
      WHERE
        /* %JoinFKPK(deleted,PlateWireframeVideo," = "," AND") */
        deleted.PlateWireframeVideoKey = PlateWireframeVideo.PlateWireframeVideoKey AND
        NOT EXISTS (
          SELECT * FROM FeaturesPerPlateWireframe
          WHERE
            /* %JoinFKPK(FeaturesPerPlateWireframe,PlateWireframeVideo," = "," AND") */
            FeaturesPerPlateWireframe.PlateWireframeVideoKey = PlateWireframeVideo.PlateWireframeVideoKey
        )
    )
    BEGIN
      SELECT @errno  = 30010,
             @errmsg = 'Cannot delete last FeaturesPerPlateWireframe because PlateWireframeVideo exists.'
      GOTO error
    END

    /* ERwin Builtin Trigger */
    /* PlateFeature  FeaturesPerPlateWireframe on child delete no action */
    /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="PlateFeature"
    CHILD_OWNER="", CHILD_TABLE="FeaturesPerPlateWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_35", FK_COLUMNS="PlateFeatureKey" */
    IF EXISTS (SELECT * FROM deleted,PlateFeature
      WHERE
        /* %JoinFKPK(deleted,PlateFeature," = "," AND") */
        deleted.PlateFeatureKey = PlateFeature.PlateFeatureKey AND
        NOT EXISTS (
          SELECT * FROM FeaturesPerPlateWireframe
          WHERE
            /* %JoinFKPK(FeaturesPerPlateWireframe,PlateFeature," = "," AND") */
            FeaturesPerPlateWireframe.PlateFeatureKey = PlateFeature.PlateFeatureKey
        )
    )
    BEGIN
      SELECT @errno  = 30010,
             @errmsg = 'Cannot delete last FeaturesPerPlateWireframe because PlateFeature exists.'
      GOTO error
    END


    /* ERwin Builtin Trigger */
    RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 

CREATE TRIGGER tU_FeaturesPerPlateWireframe ON FeaturesPerPlateWireframe FOR UPDATE AS
/* ERwin Builtin Trigger */
/* UPDATE trigger on FeaturesPerPlateWireframe */
BEGIN
  DECLARE  @numrows int,
           @nullcnt int,
           @validcnt int,
           @insFeaturesPerPlateWireframe Key,
           @errno   int,
           @errmsg  varchar(255)

  SELECT @numrows = @@rowcount
  /* ERwin Builtin Trigger */
  /* PlateWireframeVideo  FeaturesPerPlateWireframe on child update no action */
  /* ERWIN_RELATION:CHECKSUM="000309af", PARENT_OWNER="", PARENT_TABLE="PlateWireframeVideo"
    CHILD_OWNER="", CHILD_TABLE="FeaturesPerPlateWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_36", FK_COLUMNS="PlateWireframeVideoKey" */
  IF
    /* %ChildFK(" OR",UPDATE) */
    UPDATE(PlateWireframeVideoKey)
  BEGIN
    SELECT @nullcnt = 0
    SELECT @validcnt = count(*)
      FROM inserted,PlateWireframeVideo
        WHERE
          /* %JoinFKPK(inserted,PlateWireframeVideo) */
          inserted.PlateWireframeVideoKey = PlateWireframeVideo.PlateWireframeVideoKey
    /* %NotnullFK(inserted," IS NULL","select @nullcnt = count(*) from inserted where"," AND") */
    
    IF @validcnt + @nullcnt != @numrows
    BEGIN
      SELECT @errno  = 30007,
             @errmsg = 'Cannot update FeaturesPerPlateWireframe because PlateWireframeVideo does not exist.'
      GOTO error
    END
  END

  /* ERwin Builtin Trigger */
  /* PlateFeature  FeaturesPerPlateWireframe on child update no action */
  /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="PlateFeature"
    CHILD_OWNER="", CHILD_TABLE="FeaturesPerPlateWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_35", FK_COLUMNS="PlateFeatureKey" */
  IF
    /* %ChildFK(" OR",UPDATE) */
    UPDATE(PlateFeatureKey)
  BEGIN
    SELECT @nullcnt = 0
    SELECT @validcnt = count(*)
      FROM inserted,PlateFeature
        WHERE
          /* %JoinFKPK(inserted,PlateFeature) */
          inserted.PlateFeatureKey = PlateFeature.PlateFeatureKey
    /* %NotnullFK(inserted," IS NULL","select @nullcnt = count(*) from inserted where"," AND") */
    
    IF @validcnt + @nullcnt != @numrows
    BEGIN
      SELECT @errno  = 30007,
             @errmsg = 'Cannot update FeaturesPerPlateWireframe because PlateFeature does not exist.'
      GOTO error
    END
  END


  /* ERwin Builtin Trigger */
  RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 



CREATE TRIGGER tD_HistogramsPerPlateWireframe ON HistogramsPerPlateWireframe FOR DELETE AS
/* ERwin Builtin Trigger */
/* DELETE trigger on HistogramsPerPlateWireframe */
BEGIN
  DECLARE  @errno   int,
           @errmsg  varchar(255)
    /* ERwin Builtin Trigger */
    /* PlateWireframeVideo  HistogramsPerPlateWireframe on child delete no action */
    /* ERWIN_RELATION:CHECKSUM="00019f00", PARENT_OWNER="", PARENT_TABLE="PlateWireframeVideo"
    CHILD_OWNER="", CHILD_TABLE="HistogramsPerPlateWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_41", FK_COLUMNS="PlateWireframeVideoKey" */
    IF EXISTS (SELECT * FROM deleted,PlateWireframeVideo
      WHERE
        /* %JoinFKPK(deleted,PlateWireframeVideo," = "," AND") */
        deleted.PlateWireframeVideoKey = PlateWireframeVideo.PlateWireframeVideoKey AND
        NOT EXISTS (
          SELECT * FROM HistogramsPerPlateWireframe
          WHERE
            /* %JoinFKPK(HistogramsPerPlateWireframe,PlateWireframeVideo," = "," AND") */
            HistogramsPerPlateWireframe.PlateWireframeVideoKey = PlateWireframeVideo.PlateWireframeVideoKey
        )
    )
    BEGIN
      SELECT @errno  = 30010,
             @errmsg = 'Cannot delete last HistogramsPerPlateWireframe because PlateWireframeVideo exists.'
      GOTO error
    END


    /* ERwin Builtin Trigger */
    RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 

CREATE TRIGGER tU_HistogramsPerPlateWireframe ON HistogramsPerPlateWireframe FOR UPDATE AS
/* ERwin Builtin Trigger */
/* UPDATE trigger on HistogramsPerPlateWireframe */
BEGIN
  DECLARE  @numrows int,
           @nullcnt int,
           @validcnt int,
           @insHistogramsPerPlateWireframeKey Key,
           @errno   int,
           @errmsg  varchar(255)

  SELECT @numrows = @@rowcount
  /* ERwin Builtin Trigger */
  /* PlateWireframeVideo  HistogramsPerPlateWireframe on child update no action */
  /* ERWIN_RELATION:CHECKSUM="000199bb", PARENT_OWNER="", PARENT_TABLE="PlateWireframeVideo"
    CHILD_OWNER="", CHILD_TABLE="HistogramsPerPlateWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_41", FK_COLUMNS="PlateWireframeVideoKey" */
  IF
    /* %ChildFK(" OR",UPDATE) */
    UPDATE(PlateWireframeVideoKey)
  BEGIN
    SELECT @nullcnt = 0
    SELECT @validcnt = count(*)
      FROM inserted,PlateWireframeVideo
        WHERE
          /* %JoinFKPK(inserted,PlateWireframeVideo) */
          inserted.PlateWireframeVideoKey = PlateWireframeVideo.PlateWireframeVideoKey
    /* %NotnullFK(inserted," IS NULL","select @nullcnt = count(*) from inserted where"," AND") */
    
    IF @validcnt + @nullcnt != @numrows
    BEGIN
      SELECT @errno  = 30007,
             @errmsg = 'Cannot update HistogramsPerPlateWireframe because PlateWireframeVideo does not exist.'
      GOTO error
    END
  END


  /* ERwin Builtin Trigger */
  RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 



CREATE TRIGGER tD_WormWireframeVideo ON WormWireframeVideo FOR DELETE AS
/* ERwin Builtin Trigger */
/* DELETE trigger on WormWireframeVideo */
BEGIN
  DECLARE  @errno   int,
           @errmsg  varchar(255)
    /* ERwin Builtin Trigger */
    /* WormWireframeVideo  HistogramsPerWormWireframe on parent delete no action */
    /* ERWIN_RELATION:CHECKSUM="000532df", PARENT_OWNER="", PARENT_TABLE="WormWireframeVideo"
    CHILD_OWNER="", CHILD_TABLE="HistogramsPerWormWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_38", FK_COLUMNS="WormWireframeKey" */
    IF EXISTS (
      SELECT * FROM deleted,HistogramsPerWormWireframe
      WHERE
        /*  %JoinFKPK(HistogramsPerWormWireframe,deleted," = "," AND") */
        HistogramsPerWormWireframe.WormWireframeKey = deleted.WormWireframeKey
    )
    BEGIN
      SELECT @errno  = 30001,
             @errmsg = 'Cannot delete WormWireframeVideo because HistogramsPerWormWireframe exists.'
      GOTO error
    END

    /* ERwin Builtin Trigger */
    /* WormWireframeVideo  MeasurementsPerWormWireframe on parent delete no action */
    /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="WormWireframeVideo"
    CHILD_OWNER="", CHILD_TABLE="MeasurementsPerWormWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_37", FK_COLUMNS="WormWireframeKey" */
    IF EXISTS (
      SELECT * FROM deleted,MeasurementsPerWormWireframe
      WHERE
        /*  %JoinFKPK(MeasurementsPerWormWireframe,deleted," = "," AND") */
        MeasurementsPerWormWireframe.WormWireframeKey = deleted.WormWireframeKey
    )
    BEGIN
      SELECT @errno  = 30001,
             @errmsg = 'Cannot delete WormWireframeVideo because MeasurementsPerWormWireframe exists.'
      GOTO error
    END

    /* ERwin Builtin Trigger */
    /* WormWireframeVideo  FeaturesPerWormWireframe on parent delete no action */
    /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="WormWireframeVideo"
    CHILD_OWNER="", CHILD_TABLE="FeaturesPerWormWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_34", FK_COLUMNS="WormWireframeKey" */
    IF EXISTS (
      SELECT * FROM deleted,FeaturesPerWormWireframe
      WHERE
        /*  %JoinFKPK(FeaturesPerWormWireframe,deleted," = "," AND") */
        FeaturesPerWormWireframe.WormWireframeKey = deleted.WormWireframeKey
    )
    BEGIN
      SELECT @errno  = 30001,
             @errmsg = 'Cannot delete WormWireframeVideo because FeaturesPerWormWireframe exists.'
      GOTO error
    END

    /* ERwin Builtin Trigger */
    /* PlateWireframeVideo  WormWireframeVideo on child delete no action */
    /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="PlateWireframeVideo"
    CHILD_OWNER="", CHILD_TABLE="WormWireframeVideo"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_40", FK_COLUMNS="PlateWireframeVideoKey" */
    IF EXISTS (SELECT * FROM deleted,PlateWireframeVideo
      WHERE
        /* %JoinFKPK(deleted,PlateWireframeVideo," = "," AND") */
        deleted.PlateWireframeVideoKey = PlateWireframeVideo.PlateWireframeVideoKey AND
        NOT EXISTS (
          SELECT * FROM WormWireframeVideo
          WHERE
            /* %JoinFKPK(WormWireframeVideo,PlateWireframeVideo," = "," AND") */
            WormWireframeVideo.PlateWireframeVideoKey = PlateWireframeVideo.PlateWireframeVideoKey
        )
    )
    BEGIN
      SELECT @errno  = 30010,
             @errmsg = 'Cannot delete last WormWireframeVideo because PlateWireframeVideo exists.'
      GOTO error
    END


    /* ERwin Builtin Trigger */
    RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 

CREATE TRIGGER tU_WormWireframeVideo ON WormWireframeVideo FOR UPDATE AS
/* ERwin Builtin Trigger */
/* UPDATE trigger on WormWireframeVideo */
BEGIN
  DECLARE  @numrows int,
           @nullcnt int,
           @validcnt int,
           @insWormWireframeKey Key,
           @errno   int,
           @errmsg  varchar(255)

  SELECT @numrows = @@rowcount
  /* ERwin Builtin Trigger */
  /* WormWireframeVideo  HistogramsPerWormWireframe on parent update no action */
  /* ERWIN_RELATION:CHECKSUM="000563eb", PARENT_OWNER="", PARENT_TABLE="WormWireframeVideo"
    CHILD_OWNER="", CHILD_TABLE="HistogramsPerWormWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_38", FK_COLUMNS="WormWireframeKey" */
  IF
    /* %ParentPK(" OR",UPDATE) */
    UPDATE(WormWireframeKey)
  BEGIN
    IF EXISTS (
      SELECT * FROM deleted,HistogramsPerWormWireframe
      WHERE
        /*  %JoinFKPK(HistogramsPerWormWireframe,deleted," = "," AND") */
        HistogramsPerWormWireframe.WormWireframeKey = deleted.WormWireframeKey
    )
    BEGIN
      SELECT @errno  = 30005,
             @errmsg = 'Cannot update WormWireframeVideo because HistogramsPerWormWireframe exists.'
      GOTO error
    END
  END

  /* ERwin Builtin Trigger */
  /* WormWireframeVideo  MeasurementsPerWormWireframe on parent update no action */
  /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="WormWireframeVideo"
    CHILD_OWNER="", CHILD_TABLE="MeasurementsPerWormWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_37", FK_COLUMNS="WormWireframeKey" */
  IF
    /* %ParentPK(" OR",UPDATE) */
    UPDATE(WormWireframeKey)
  BEGIN
    IF EXISTS (
      SELECT * FROM deleted,MeasurementsPerWormWireframe
      WHERE
        /*  %JoinFKPK(MeasurementsPerWormWireframe,deleted," = "," AND") */
        MeasurementsPerWormWireframe.WormWireframeKey = deleted.WormWireframeKey
    )
    BEGIN
      SELECT @errno  = 30005,
             @errmsg = 'Cannot update WormWireframeVideo because MeasurementsPerWormWireframe exists.'
      GOTO error
    END
  END

  /* ERwin Builtin Trigger */
  /* WormWireframeVideo  FeaturesPerWormWireframe on parent update no action */
  /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="WormWireframeVideo"
    CHILD_OWNER="", CHILD_TABLE="FeaturesPerWormWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_34", FK_COLUMNS="WormWireframeKey" */
  IF
    /* %ParentPK(" OR",UPDATE) */
    UPDATE(WormWireframeKey)
  BEGIN
    IF EXISTS (
      SELECT * FROM deleted,FeaturesPerWormWireframe
      WHERE
        /*  %JoinFKPK(FeaturesPerWormWireframe,deleted," = "," AND") */
        FeaturesPerWormWireframe.WormWireframeKey = deleted.WormWireframeKey
    )
    BEGIN
      SELECT @errno  = 30005,
             @errmsg = 'Cannot update WormWireframeVideo because FeaturesPerWormWireframe exists.'
      GOTO error
    END
  END

  /* ERwin Builtin Trigger */
  /* PlateWireframeVideo  WormWireframeVideo on child update no action */
  /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="PlateWireframeVideo"
    CHILD_OWNER="", CHILD_TABLE="WormWireframeVideo"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_40", FK_COLUMNS="PlateWireframeVideoKey" */
  IF
    /* %ChildFK(" OR",UPDATE) */
    UPDATE(PlateWireframeVideoKey)
  BEGIN
    SELECT @nullcnt = 0
    SELECT @validcnt = count(*)
      FROM inserted,PlateWireframeVideo
        WHERE
          /* %JoinFKPK(inserted,PlateWireframeVideo) */
          inserted.PlateWireframeVideoKey = PlateWireframeVideo.PlateWireframeVideoKey
    /* %NotnullFK(inserted," IS NULL","select @nullcnt = count(*) from inserted where"," AND") */
    
    IF @validcnt + @nullcnt != @numrows
    BEGIN
      SELECT @errno  = 30007,
             @errmsg = 'Cannot update WormWireframeVideo because PlateWireframeVideo does not exist.'
      GOTO error
    END
  END


  /* ERwin Builtin Trigger */
  RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 



CREATE TRIGGER tD_HistogramsPerWormWireframe ON HistogramsPerWormWireframe FOR DELETE AS
/* ERwin Builtin Trigger */
/* DELETE trigger on HistogramsPerWormWireframe */
BEGIN
  DECLARE  @errno   int,
           @errmsg  varchar(255)
    /* ERwin Builtin Trigger */
    /* WormWireframeVideo  HistogramsPerWormWireframe on child delete no action */
    /* ERWIN_RELATION:CHECKSUM="00059950", PARENT_OWNER="", PARENT_TABLE="WormWireframeVideo"
    CHILD_OWNER="", CHILD_TABLE="HistogramsPerWormWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_38", FK_COLUMNS="WormWireframeKey" */
    IF EXISTS (SELECT * FROM deleted,WormWireframeVideo
      WHERE
        /* %JoinFKPK(deleted,WormWireframeVideo," = "," AND") */
        deleted.WormWireframeKey = WormWireframeVideo.WormWireframeKey AND
        NOT EXISTS (
          SELECT * FROM HistogramsPerWormWireframe
          WHERE
            /* %JoinFKPK(HistogramsPerWormWireframe,WormWireframeVideo," = "," AND") */
            HistogramsPerWormWireframe.WormWireframeKey = WormWireframeVideo.WormWireframeKey
        )
    )
    BEGIN
      SELECT @errno  = 30010,
             @errmsg = 'Cannot delete last HistogramsPerWormWireframe because WormWireframeVideo exists.'
      GOTO error
    END

    /* ERwin Builtin Trigger */
    /* WormFeature  HistogramsPerWormWireframe on child delete no action */
    /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="WormFeature"
    CHILD_OWNER="", CHILD_TABLE="HistogramsPerWormWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_27", FK_COLUMNS="WormFeatureKey" */
    IF EXISTS (SELECT * FROM deleted,WormFeature
      WHERE
        /* %JoinFKPK(deleted,WormFeature," = "," AND") */
        deleted.WormFeatureKey = WormFeature.WormFeatureKey AND
        NOT EXISTS (
          SELECT * FROM HistogramsPerWormWireframe
          WHERE
            /* %JoinFKPK(HistogramsPerWormWireframe,WormFeature," = "," AND") */
            HistogramsPerWormWireframe.WormFeatureKey = WormFeature.WormFeatureKey
        )
    )
    BEGIN
      SELECT @errno  = 30010,
             @errmsg = 'Cannot delete last HistogramsPerWormWireframe because WormFeature exists.'
      GOTO error
    END

    /* ERwin Builtin Trigger */
    /* Direction  HistogramsPerWormWireframe on child delete no action */
    /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="Direction"
    CHILD_OWNER="", CHILD_TABLE="HistogramsPerWormWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_17", FK_COLUMNS="EventDirectionKey" */
    IF EXISTS (SELECT * FROM deleted,Direction
      WHERE
        /* %JoinFKPK(deleted,Direction," = "," AND") */
        deleted.EventDirectionKey = Direction.DirectionKey AND
        NOT EXISTS (
          SELECT * FROM HistogramsPerWormWireframe
          WHERE
            /* %JoinFKPK(HistogramsPerWormWireframe,Direction," = "," AND") */
            HistogramsPerWormWireframe.EventDirectionKey = Direction.DirectionKey
        )
    )
    BEGIN
      SELECT @errno  = 30010,
             @errmsg = 'Cannot delete last HistogramsPerWormWireframe because Direction exists.'
      GOTO error
    END

    /* ERwin Builtin Trigger */
    /* Sign  HistogramsPerWormWireframe on child delete no action */
    /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="Sign"
    CHILD_OWNER="", CHILD_TABLE="HistogramsPerWormWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_16", FK_COLUMNS="SignKey" */
    IF EXISTS (SELECT * FROM deleted,Sign
      WHERE
        /* %JoinFKPK(deleted,Sign," = "," AND") */
        deleted.SignKey = Sign.SignKey AND
        NOT EXISTS (
          SELECT * FROM HistogramsPerWormWireframe
          WHERE
            /* %JoinFKPK(HistogramsPerWormWireframe,Sign," = "," AND") */
            HistogramsPerWormWireframe.SignKey = Sign.SignKey
        )
    )
    BEGIN
      SELECT @errno  = 30010,
             @errmsg = 'Cannot delete last HistogramsPerWormWireframe because Sign exists.'
      GOTO error
    END


    /* ERwin Builtin Trigger */
    RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 

CREATE TRIGGER tU_HistogramsPerWormWireframe ON HistogramsPerWormWireframe FOR UPDATE AS
/* ERwin Builtin Trigger */
/* UPDATE trigger on HistogramsPerWormWireframe */
BEGIN
  DECLARE  @numrows int,
           @nullcnt int,
           @validcnt int,
           @insHistogramsPerWormWireframeKey Key,
           @errno   int,
           @errmsg  varchar(255)

  SELECT @numrows = @@rowcount
  /* ERwin Builtin Trigger */
  /* WormWireframeVideo  HistogramsPerWormWireframe on child update no action */
  /* ERWIN_RELATION:CHECKSUM="0005a5bb", PARENT_OWNER="", PARENT_TABLE="WormWireframeVideo"
    CHILD_OWNER="", CHILD_TABLE="HistogramsPerWormWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_38", FK_COLUMNS="WormWireframeKey" */
  IF
    /* %ChildFK(" OR",UPDATE) */
    UPDATE(WormWireframeKey)
  BEGIN
    SELECT @nullcnt = 0
    SELECT @validcnt = count(*)
      FROM inserted,WormWireframeVideo
        WHERE
          /* %JoinFKPK(inserted,WormWireframeVideo) */
          inserted.WormWireframeKey = WormWireframeVideo.WormWireframeKey
    /* %NotnullFK(inserted," IS NULL","select @nullcnt = count(*) from inserted where"," AND") */
    
    IF @validcnt + @nullcnt != @numrows
    BEGIN
      SELECT @errno  = 30007,
             @errmsg = 'Cannot update HistogramsPerWormWireframe because WormWireframeVideo does not exist.'
      GOTO error
    END
  END

  /* ERwin Builtin Trigger */
  /* WormFeature  HistogramsPerWormWireframe on child update no action */
  /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="WormFeature"
    CHILD_OWNER="", CHILD_TABLE="HistogramsPerWormWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_27", FK_COLUMNS="WormFeatureKey" */
  IF
    /* %ChildFK(" OR",UPDATE) */
    UPDATE(WormFeatureKey)
  BEGIN
    SELECT @nullcnt = 0
    SELECT @validcnt = count(*)
      FROM inserted,WormFeature
        WHERE
          /* %JoinFKPK(inserted,WormFeature) */
          inserted.WormFeatureKey = WormFeature.WormFeatureKey
    /* %NotnullFK(inserted," IS NULL","select @nullcnt = count(*) from inserted where"," AND") */
    
    IF @validcnt + @nullcnt != @numrows
    BEGIN
      SELECT @errno  = 30007,
             @errmsg = 'Cannot update HistogramsPerWormWireframe because WormFeature does not exist.'
      GOTO error
    END
  END

  /* ERwin Builtin Trigger */
  /* Direction  HistogramsPerWormWireframe on child update no action */
  /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="Direction"
    CHILD_OWNER="", CHILD_TABLE="HistogramsPerWormWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_17", FK_COLUMNS="EventDirectionKey" */
  IF
    /* %ChildFK(" OR",UPDATE) */
    UPDATE(EventDirectionKey)
  BEGIN
    SELECT @nullcnt = 0
    SELECT @validcnt = count(*)
      FROM inserted,Direction
        WHERE
          /* %JoinFKPK(inserted,Direction) */
          inserted.EventDirectionKey = Direction.DirectionKey
    /* %NotnullFK(inserted," IS NULL","select @nullcnt = count(*) from inserted where"," AND") */
    
    IF @validcnt + @nullcnt != @numrows
    BEGIN
      SELECT @errno  = 30007,
             @errmsg = 'Cannot update HistogramsPerWormWireframe because Direction does not exist.'
      GOTO error
    END
  END

  /* ERwin Builtin Trigger */
  /* Sign  HistogramsPerWormWireframe on child update no action */
  /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="Sign"
    CHILD_OWNER="", CHILD_TABLE="HistogramsPerWormWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_16", FK_COLUMNS="SignKey" */
  IF
    /* %ChildFK(" OR",UPDATE) */
    UPDATE(SignKey)
  BEGIN
    SELECT @nullcnt = 0
    SELECT @validcnt = count(*)
      FROM inserted,Sign
        WHERE
          /* %JoinFKPK(inserted,Sign) */
          inserted.SignKey = Sign.SignKey
    /* %NotnullFK(inserted," IS NULL","select @nullcnt = count(*) from inserted where"," AND") */
    
    IF @validcnt + @nullcnt != @numrows
    BEGIN
      SELECT @errno  = 30007,
             @errmsg = 'Cannot update HistogramsPerWormWireframe because Sign does not exist.'
      GOTO error
    END
  END


  /* ERwin Builtin Trigger */
  RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 



CREATE TRIGGER tD_FeaturesPerWormWireframe ON FeaturesPerWormWireframe FOR DELETE AS
/* ERwin Builtin Trigger */
/* DELETE trigger on FeaturesPerWormWireframe */
BEGIN
  DECLARE  @errno   int,
           @errmsg  varchar(255)
    /* ERwin Builtin Trigger */
    /* WormWireframeVideo  FeaturesPerWormWireframe on child delete no action */
    /* ERWIN_RELATION:CHECKSUM="0002f85e", PARENT_OWNER="", PARENT_TABLE="WormWireframeVideo"
    CHILD_OWNER="", CHILD_TABLE="FeaturesPerWormWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_34", FK_COLUMNS="WormWireframeKey" */
    IF EXISTS (SELECT * FROM deleted,WormWireframeVideo
      WHERE
        /* %JoinFKPK(deleted,WormWireframeVideo," = "," AND") */
        deleted.WormWireframeKey = WormWireframeVideo.WormWireframeKey AND
        NOT EXISTS (
          SELECT * FROM FeaturesPerWormWireframe
          WHERE
            /* %JoinFKPK(FeaturesPerWormWireframe,WormWireframeVideo," = "," AND") */
            FeaturesPerWormWireframe.WormWireframeKey = WormWireframeVideo.WormWireframeKey
        )
    )
    BEGIN
      SELECT @errno  = 30010,
             @errmsg = 'Cannot delete last FeaturesPerWormWireframe because WormWireframeVideo exists.'
      GOTO error
    END

    /* ERwin Builtin Trigger */
    /* WormFeature  FeaturesPerWormWireframe on child delete no action */
    /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="WormFeature"
    CHILD_OWNER="", CHILD_TABLE="FeaturesPerWormWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_20", FK_COLUMNS="WormFeatureKey" */
    IF EXISTS (SELECT * FROM deleted,WormFeature
      WHERE
        /* %JoinFKPK(deleted,WormFeature," = "," AND") */
        deleted.WormFeatureKey = WormFeature.WormFeatureKey AND
        NOT EXISTS (
          SELECT * FROM FeaturesPerWormWireframe
          WHERE
            /* %JoinFKPK(FeaturesPerWormWireframe,WormFeature," = "," AND") */
            FeaturesPerWormWireframe.WormFeatureKey = WormFeature.WormFeatureKey
        )
    )
    BEGIN
      SELECT @errno  = 30010,
             @errmsg = 'Cannot delete last FeaturesPerWormWireframe because WormFeature exists.'
      GOTO error
    END


    /* ERwin Builtin Trigger */
    RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 

CREATE TRIGGER tU_FeaturesPerWormWireframe ON FeaturesPerWormWireframe FOR UPDATE AS
/* ERwin Builtin Trigger */
/* UPDATE trigger on FeaturesPerWormWireframe */
BEGIN
  DECLARE  @numrows int,
           @nullcnt int,
           @validcnt int,
           @insFeaturesPerWormWireframeKey Key,
           @errno   int,
           @errmsg  varchar(255)

  SELECT @numrows = @@rowcount
  /* ERwin Builtin Trigger */
  /* WormWireframeVideo  FeaturesPerWormWireframe on child update no action */
  /* ERWIN_RELATION:CHECKSUM="0002eda5", PARENT_OWNER="", PARENT_TABLE="WormWireframeVideo"
    CHILD_OWNER="", CHILD_TABLE="FeaturesPerWormWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_34", FK_COLUMNS="WormWireframeKey" */
  IF
    /* %ChildFK(" OR",UPDATE) */
    UPDATE(WormWireframeKey)
  BEGIN
    SELECT @nullcnt = 0
    SELECT @validcnt = count(*)
      FROM inserted,WormWireframeVideo
        WHERE
          /* %JoinFKPK(inserted,WormWireframeVideo) */
          inserted.WormWireframeKey = WormWireframeVideo.WormWireframeKey
    /* %NotnullFK(inserted," IS NULL","select @nullcnt = count(*) from inserted where"," AND") */
    
    IF @validcnt + @nullcnt != @numrows
    BEGIN
      SELECT @errno  = 30007,
             @errmsg = 'Cannot update FeaturesPerWormWireframe because WormWireframeVideo does not exist.'
      GOTO error
    END
  END

  /* ERwin Builtin Trigger */
  /* WormFeature  FeaturesPerWormWireframe on child update no action */
  /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="WormFeature"
    CHILD_OWNER="", CHILD_TABLE="FeaturesPerWormWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_20", FK_COLUMNS="WormFeatureKey" */
  IF
    /* %ChildFK(" OR",UPDATE) */
    UPDATE(WormFeatureKey)
  BEGIN
    SELECT @nullcnt = 0
    SELECT @validcnt = count(*)
      FROM inserted,WormFeature
        WHERE
          /* %JoinFKPK(inserted,WormFeature) */
          inserted.WormFeatureKey = WormFeature.WormFeatureKey
    /* %NotnullFK(inserted," IS NULL","select @nullcnt = count(*) from inserted where"," AND") */
    
    IF @validcnt + @nullcnt != @numrows
    BEGIN
      SELECT @errno  = 30007,
             @errmsg = 'Cannot update FeaturesPerWormWireframe because WormFeature does not exist.'
      GOTO error
    END
  END


  /* ERwin Builtin Trigger */
  RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 



CREATE TRIGGER tD_WormMeasurement ON WormMeasurement FOR DELETE AS
/* ERwin Builtin Trigger */
/* DELETE trigger on WormMeasurement */
BEGIN
  DECLARE  @errno   int,
           @errmsg  varchar(255)
    /* ERwin Builtin Trigger */
    /* WormMeasurement  MeasurementsPerWormWireframe on parent delete no action */
    /* ERWIN_RELATION:CHECKSUM="000139ca", PARENT_OWNER="", PARENT_TABLE="WormMeasurement"
    CHILD_OWNER="", CHILD_TABLE="MeasurementsPerWormWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_23", FK_COLUMNS="WormMeasurementsKey" */
    IF EXISTS (
      SELECT * FROM deleted,MeasurementsPerWormWireframe
      WHERE
        /*  %JoinFKPK(MeasurementsPerWormWireframe,deleted," = "," AND") */
        MeasurementsPerWormWireframe.WormMeasurementsKey = deleted.WormMeasurementsKey
    )
    BEGIN
      SELECT @errno  = 30001,
             @errmsg = 'Cannot delete WormMeasurement because MeasurementsPerWormWireframe exists.'
      GOTO error
    END


    /* ERwin Builtin Trigger */
    RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 

CREATE TRIGGER tU_WormMeasurement ON WormMeasurement FOR UPDATE AS
/* ERwin Builtin Trigger */
/* UPDATE trigger on WormMeasurement */
BEGIN
  DECLARE  @numrows int,
           @nullcnt int,
           @validcnt int,
           @insWormMeasurementsKey Key,
           @errno   int,
           @errmsg  varchar(255)

  SELECT @numrows = @@rowcount
  /* ERwin Builtin Trigger */
  /* WormMeasurement  MeasurementsPerWormWireframe on parent update no action */
  /* ERWIN_RELATION:CHECKSUM="000150b5", PARENT_OWNER="", PARENT_TABLE="WormMeasurement"
    CHILD_OWNER="", CHILD_TABLE="MeasurementsPerWormWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_23", FK_COLUMNS="WormMeasurementsKey" */
  IF
    /* %ParentPK(" OR",UPDATE) */
    UPDATE(WormMeasurementsKey)
  BEGIN
    IF EXISTS (
      SELECT * FROM deleted,MeasurementsPerWormWireframe
      WHERE
        /*  %JoinFKPK(MeasurementsPerWormWireframe,deleted," = "," AND") */
        MeasurementsPerWormWireframe.WormMeasurementsKey = deleted.WormMeasurementsKey
    )
    BEGIN
      SELECT @errno  = 30005,
             @errmsg = 'Cannot update WormMeasurement because MeasurementsPerWormWireframe exists.'
      GOTO error
    END
  END


  /* ERwin Builtin Trigger */
  RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 



CREATE TRIGGER tD_MeasurementsPerWormWireframe ON MeasurementsPerWormWireframe FOR DELETE AS
/* ERwin Builtin Trigger */
/* DELETE trigger on MeasurementsPerWormWireframe */
BEGIN
  DECLARE  @errno   int,
           @errmsg  varchar(255)
    /* ERwin Builtin Trigger */
    /* WormWireframeVideo  MeasurementsPerWormWireframe on child delete no action */
    /* ERWIN_RELATION:CHECKSUM="00032b9e", PARENT_OWNER="", PARENT_TABLE="WormWireframeVideo"
    CHILD_OWNER="", CHILD_TABLE="MeasurementsPerWormWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_37", FK_COLUMNS="WormWireframeKey" */
    IF EXISTS (SELECT * FROM deleted,WormWireframeVideo
      WHERE
        /* %JoinFKPK(deleted,WormWireframeVideo," = "," AND") */
        deleted.WormWireframeKey = WormWireframeVideo.WormWireframeKey AND
        NOT EXISTS (
          SELECT * FROM MeasurementsPerWormWireframe
          WHERE
            /* %JoinFKPK(MeasurementsPerWormWireframe,WormWireframeVideo," = "," AND") */
            MeasurementsPerWormWireframe.WormWireframeKey = WormWireframeVideo.WormWireframeKey
        )
    )
    BEGIN
      SELECT @errno  = 30010,
             @errmsg = 'Cannot delete last MeasurementsPerWormWireframe because WormWireframeVideo exists.'
      GOTO error
    END

    /* ERwin Builtin Trigger */
    /* WormMeasurement  MeasurementsPerWormWireframe on child delete no action */
    /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="WormMeasurement"
    CHILD_OWNER="", CHILD_TABLE="MeasurementsPerWormWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_23", FK_COLUMNS="WormMeasurementsKey" */
    IF EXISTS (SELECT * FROM deleted,WormMeasurement
      WHERE
        /* %JoinFKPK(deleted,WormMeasurement," = "," AND") */
        deleted.WormMeasurementsKey = WormMeasurement.WormMeasurementsKey AND
        NOT EXISTS (
          SELECT * FROM MeasurementsPerWormWireframe
          WHERE
            /* %JoinFKPK(MeasurementsPerWormWireframe,WormMeasurement," = "," AND") */
            MeasurementsPerWormWireframe.WormMeasurementsKey = WormMeasurement.WormMeasurementsKey
        )
    )
    BEGIN
      SELECT @errno  = 30010,
             @errmsg = 'Cannot delete last MeasurementsPerWormWireframe because WormMeasurement exists.'
      GOTO error
    END


    /* ERwin Builtin Trigger */
    RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 

CREATE TRIGGER tU_MeasurementsPerWormWireframe ON MeasurementsPerWormWireframe FOR UPDATE AS
/* ERwin Builtin Trigger */
/* UPDATE trigger on MeasurementsPerWormWireframe */
BEGIN
  DECLARE  @numrows int,
           @nullcnt int,
           @validcnt int,
           @insMeasurementsPerWormWireframe Key,
           @errno   int,
           @errmsg  varchar(255)

  SELECT @numrows = @@rowcount
  /* ERwin Builtin Trigger */
  /* WormWireframeVideo  MeasurementsPerWormWireframe on child update no action */
  /* ERWIN_RELATION:CHECKSUM="00030c12", PARENT_OWNER="", PARENT_TABLE="WormWireframeVideo"
    CHILD_OWNER="", CHILD_TABLE="MeasurementsPerWormWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_37", FK_COLUMNS="WormWireframeKey" */
  IF
    /* %ChildFK(" OR",UPDATE) */
    UPDATE(WormWireframeKey)
  BEGIN
    SELECT @nullcnt = 0
    SELECT @validcnt = count(*)
      FROM inserted,WormWireframeVideo
        WHERE
          /* %JoinFKPK(inserted,WormWireframeVideo) */
          inserted.WormWireframeKey = WormWireframeVideo.WormWireframeKey
    /* %NotnullFK(inserted," IS NULL","select @nullcnt = count(*) from inserted where"," AND") */
    
    IF @validcnt + @nullcnt != @numrows
    BEGIN
      SELECT @errno  = 30007,
             @errmsg = 'Cannot update MeasurementsPerWormWireframe because WormWireframeVideo does not exist.'
      GOTO error
    END
  END

  /* ERwin Builtin Trigger */
  /* WormMeasurement  MeasurementsPerWormWireframe on child update no action */
  /* ERWIN_RELATION:CHECKSUM="00000000", PARENT_OWNER="", PARENT_TABLE="WormMeasurement"
    CHILD_OWNER="", CHILD_TABLE="MeasurementsPerWormWireframe"
    P2C_VERB_PHRASE="", C2P_VERB_PHRASE="", 
    FK_CONSTRAINT="R_23", FK_COLUMNS="WormMeasurementsKey" */
  IF
    /* %ChildFK(" OR",UPDATE) */
    UPDATE(WormMeasurementsKey)
  BEGIN
    SELECT @nullcnt = 0
    SELECT @validcnt = count(*)
      FROM inserted,WormMeasurement
        WHERE
          /* %JoinFKPK(inserted,WormMeasurement) */
          inserted.WormMeasurementsKey = WormMeasurement.WormMeasurementsKey
    /* %NotnullFK(inserted," IS NULL","select @nullcnt = count(*) from inserted where"," AND") */
    
    IF @validcnt + @nullcnt != @numrows
    BEGIN
      SELECT @errno  = 30007,
             @errmsg = 'Cannot update MeasurementsPerWormWireframe because WormMeasurement does not exist.'
      GOTO error
    END
  END


  /* ERwin Builtin Trigger */
  RETURN
error:
    raiserror @errno @errmsg
    rollback transaction
END

go
 

