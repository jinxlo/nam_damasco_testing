--
-- PostgreSQL database dump
--

-- Dumped from database version 15.12 (Debian 15.12-1.pgdg120+1)
-- Dumped by pg_dump version 15.12 (Debian 15.12-1.pgdg120+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: unaccent; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS unaccent WITH SCHEMA public;


--
-- Name: EXTENSION unaccent; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION unaccent IS 'text search dictionary that removes accents';


--
-- Name: vector; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;


--
-- Name: EXTENSION vector; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION vector IS 'vector data type and ivfflat and hnsw access methods';


--
-- Name: canonicalize_whs(text); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.canonicalize_whs(text) RETURNS text
    LANGUAGE sql IMMUTABLE STRICT
    AS $_$
SELECT upper(
         regexp_replace(
           unaccent($1),
           '[^A-Za-z0-9]+',      -- anything that is NOT A-Z / a-z / 0-9
           '_',                  -- replace with single underscore
           'g'                   -- global
         )
       );
$_$;


--
-- Name: trg_set_canonical_whs(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.trg_set_canonical_whs() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    -- Recompute canonical warehouse name
    NEW.warehouse_name_canonical :=
        canonicalize_whs(NEW.warehouse_name);

    -- Re-issue the composite id (trim at 512 chars to stay in sync with app logic)
    NEW.id :=
        LEFT(NEW.item_code || '_' || NEW.warehouse_name_canonical, 512);

    RETURN NEW;
END;
$$;


--
-- Name: trigger_set_timestamp(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.trigger_set_timestamp() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$;


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: conversation_pauses; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.conversation_pauses (
    conversation_id character varying(255) NOT NULL,
    paused_until timestamp with time zone NOT NULL
);


--
-- Name: products; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.products (
    id character varying(512) NOT NULL,
    item_code character varying(64) NOT NULL,
    item_name text NOT NULL,
    category character varying(128),
    sub_category character varying(128),
    brand character varying(128),
    line character varying(128),
    item_group_name character varying(128),
    warehouse_name character varying(255) NOT NULL,
    branch_name character varying(255),
    price numeric(12,2),
    stock integer DEFAULT 0,
    searchable_text_content text,
    embedding public.vector(1536),
    source_data_json jsonb DEFAULT '{}'::jsonb,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    description text,
    llm_summarized_description text,
    price_bolivar numeric(12,2),
    warehouse_name_canonical character varying(255) NOT NULL,
    specifitacion text,
    store_address text
);


--
-- Name: TABLE products; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.products IS 'Stores specific product stock entries at each warehouse, synchronized from Damasco API, including vector embeddings for semantic search of the product description.';


--
-- Name: COLUMN products.id; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.products.id IS 'Application-generated composite PK: item_code + sanitized warehouse_name.';


--
-- Name: COLUMN products.warehouse_name; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.products.warehouse_name IS 'The specific warehouse where this stock entry is located (from Damasco whsName).';


--
-- Name: COLUMN products.embedding; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.products.embedding IS 'Vector embedding generated from product descriptive text (brand, name, category, etc.).';


--
-- Name: test_table; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.test_table (
    id integer
);


--
-- Name: conversation_pauses conversation_pauses_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.conversation_pauses
    ADD CONSTRAINT conversation_pauses_pkey PRIMARY KEY (conversation_id);


--
-- Name: products products_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.products
    ADD CONSTRAINT products_pkey PRIMARY KEY (id);


--
-- Name: idx_products_branch_name; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_products_branch_name ON public.products USING btree (branch_name);


--
-- Name: idx_products_brand; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_products_brand ON public.products USING btree (brand);


--
-- Name: idx_products_category; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_products_category ON public.products USING btree (category);


--
-- Name: idx_products_embedding_hnsw; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_products_embedding_hnsw ON public.products USING hnsw (embedding public.vector_cosine_ops);


--
-- Name: idx_products_item_code; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_products_item_code ON public.products USING btree (item_code);


--
-- Name: idx_products_warehouse_name; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_products_warehouse_name ON public.products USING btree (warehouse_name);


--
-- Name: idx_products_warehouse_name_canonical; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_products_warehouse_name_canonical ON public.products USING btree (warehouse_name_canonical);


--
-- Name: uq_item_code_per_whs_canonical; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX uq_item_code_per_whs_canonical ON public.products USING btree (lower((item_code)::text), warehouse_name_canonical);


--
-- Name: products set_products_timestamp; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER set_products_timestamp BEFORE UPDATE ON public.products FOR EACH ROW EXECUTE FUNCTION public.trigger_set_timestamp();


--
-- Name: products trg_products_canonical_whs; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER trg_products_canonical_whs BEFORE INSERT OR UPDATE OF warehouse_name, item_code ON public.products FOR EACH ROW EXECUTE FUNCTION public.trg_set_canonical_whs();


--
-- PostgreSQL database dump complete
--

