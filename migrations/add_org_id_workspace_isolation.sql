-- Migration: Add org_id to practice_assessments and topic_mastery for workspace isolation
-- Run this in your Supabase SQL editor or via psql before deploying the updated backend.
-- All existing rows will default to NULL (personal workspace).

-- 1. Add org_id column to practice_assessments
ALTER TABLE practice_assessments
  ADD COLUMN IF NOT EXISTS org_id UUID REFERENCES organizations(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_practice_assessments_org_id
  ON practice_assessments(org_id);

-- 2. Add org_id column to topic_mastery
ALTER TABLE topic_mastery
  ADD COLUMN IF NOT EXISTS org_id UUID REFERENCES organizations(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_topic_mastery_org_id
  ON topic_mastery(org_id);
